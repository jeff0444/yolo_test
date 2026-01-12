from ultralytics import YOLO
import numpy as np
import cv2
import time
from pathlib import Path
import torch
import argparse
import tqdm
import threading
import os
import sys
from saveStream_ffmpeg import SaveStream_ffmpeg
import collections

class DetectionProgressBar:
    def __init__(self, total_frames, height=5):
        self.total_frames = total_frames
        self.height = height
        self.detection_ranges = [] # list of (start_idx, end_idx)
        self.current_detection_start = None
        self.frame_count = 0
        self.color_normal = (0, 165, 255) # Orange (BGR)
        self.color_detect = (0, 0, 255)   # Red (BGR)

    def update(self, has_detection):
        self.frame_count += 1
        idx = self.frame_count - 1

        if has_detection:
            if self.current_detection_start is None:
                self.current_detection_start = idx
        else:
            if self.current_detection_start is not None:
                self.detection_ranges.append((self.current_detection_start, idx - 1))
                self.current_detection_start = None

    def draw(self, image):
        if self.total_frames <= 0:
            return image

        h, w = image.shape[:2]
        bar_y = h - self.height

        # Calculate current progress width
        # Ensure we don't go out of bounds if frame_count exceeds total_frames (e.g. estimate wrong)
        safe_frame_count = min(self.frame_count, self.total_frames)
        progress_width = int((safe_frame_count / self.total_frames) * w)

        # Draw Orange Bar
        cv2.rectangle(image, (0, bar_y), (progress_width, h), self.color_normal, -1)

        # Close current range for drawing purposes (if active)
        ranges_to_draw = list(self.detection_ranges)
        if self.current_detection_start is not None:
             ranges_to_draw.append((self.current_detection_start, self.frame_count - 1))

        scale = w / self.total_frames

        for start, end in ranges_to_draw:
            x1 = int(start * scale)
            x2 = int((end + 1) * scale) # +1 to include the end frame in width

            # Ensure min 3px width
            width = x2 - x1
            if width < 3:
                # Expand
                diff = 3 - width
                x1 -= diff // 2
                x2 += (diff + 1) // 2

            # Clip to image bounds
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))

            if x2 > x1:
                cv2.rectangle(image, (x1, bar_y), (x2, h), self.color_detect, -1)

        return image

class RTSPStream:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.status, self.frame = self.capture.read()
        self.stop_thread = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stop_thread:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                with self.lock:
                    self.status = status
                    self.frame = frame
                if not status:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.status, self.frame if self.frame is not None else None

    def stop(self):
        self.stop_thread = True
        self.thread.join()
        self.capture.release()

class Detector:
    def __init__(self, model_path='yolov8n.pt', device='cpu', conf_threshold=0.25, iou_threshold=0.45):
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model.to(self.device)

    def predict(self, image):
        return self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)

    def draw_results(self, image, results):
        for result in results:
            return result.plot()
        return image

def process_video(source, detector, output_path, is_rtsp=False, show=False, save_mode='all'):
    if is_rtsp:
        stream = RTSPStream(source)
        time.sleep(1.0)
        status, frame = stream.read()
        target_fps = -1 # Let Estimator calculate
        total_frames = 0
        h, w = frame.shape[:2] if frame is not None else (640, 640)
    else:
        cap = cv2.VideoCapture(source)
        fps_prop = cap.get(cv2.CAP_PROP_FPS)
        target_fps = fps_prop if fps_prop > 0 else -1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        status, frame = cap.read()
        h, w = frame.shape[:2] if frame is not None else (640, 640)

    if not status or frame is None:
        print(f"Error: Could not open source {source}")
        return

    # Initialize Saver
    saver = SaveStream_ffmpeg(output_path, fps=target_fps, width=w, height=h)

    pbar = None
    if not is_rtsp and total_frames > 0:
        pbar = tqdm.tqdm(total=total_frames, desc="Processing Video")

    # Initialize Progress Bar for visualization
    visual_pbar = DetectionProgressBar(total_frames) if (save_mode == 'all' and not is_rtsp and total_frames > 0) else None

    # Key mode buffers
    pre_event_buffer = collections.deque() # stores (frame, timestamp)
    last_written_ts = -1.0
    recording_end_time = -1.0

    try:
        while True:
            # Need strict timing for buffer logic
            current_time_sec = time.time()
            ts_ms = current_time_sec * 1000

            if is_rtsp:
                status, frame = stream.read()
                if not status:
                    time.sleep(0.01)
                    continue
            else:
                status, frame = cap.read()
                if not status:
                    break

            # Always update FPS estimator to keep it accurate to input source
            saver.update_fps(ts_ms)

            results = detector.predict(frame)
            annotated_frame = detector.draw_results(frame, results)

            if visual_pbar:
                has_obj = any(len(r.boxes) > 0 for r in results)
                visual_pbar.update(has_obj)
                annotated_frame = visual_pbar.draw(annotated_frame)

            if save_mode == 'all':
                saver.write(annotated_frame, timestamp=ts_ms, update_fps=False)

            elif save_mode == 'key':
                has_obj = False
                for result in results:
                    if len(result.boxes) > 0:
                        has_obj = True
                        break

                # Update buffer
                pre_event_buffer.append((annotated_frame, ts_ms))

                # Clean old buffer (keep > 3s)
                while len(pre_event_buffer) > 0:
                    delta_t = (ts_ms - pre_event_buffer[0][1]) / 1000.0
                    if delta_t > 3.5: # keep slightly more than 3s
                         pre_event_buffer.popleft()
                    else:
                        break

                if has_obj:
                    recording_end_time = current_time_sec + 3.0

                should_record = False
                if current_time_sec < recording_end_time:
                    should_record = True

                # Check if we need to flush buffer (did we jump into recording state?)
                # We need to ensure anything in buffer that is within [now-3s, now] AND hasn't been written is written.

                # Actually simpler: if recording_end_time is active (meaning an event happened recently),
                # we want to ensure we covered the range [event_time - 3s, event_time + 3s].
                # Since recording_end_time extends the future, we just need to look back.

                if should_record:
                    # Write frames from buffer that haven't been written
                    # Iterate copy to avoid issues if we pop (though we only pop old stuff)
                    for buf_frame, buf_ts in pre_event_buffer:
                        # Logic: Write if it's NEWER than last written
                        if buf_ts > last_written_ts:
                            saver.write(buf_frame, timestamp=buf_ts, update_fps=False)
                            last_written_ts = buf_ts

            if show:
                cv2.imshow("Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if pbar:
                pbar.update(1)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        if is_rtsp:
            stream.stop()
        else:
            cap.release()
            if pbar: pbar.close()

        saver.close()

        if show:
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Input source: image, video, directory, or RTSP URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--conf', type=float, default=0.45, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--device', type=str, default='', help='Device (cpu, cuda, 0, 1...)')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--show', action='store_true', help='Show real-time detection results')
    parser.add_argument('--save_mode', type=str, default='all', choices=['all', 'key'], help='Save mode: all or key (only with detections +/- 3s)')

    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExamples:")
        print("python detector.py --source path/to/image.jpg")
        print("python detector.py --source path/to/video.mp4")
        print("python detector.py --source path/to/images_folder")
        print("python detector.py --source rtsp://username:password@ip:port/stream")
        sys.exit(0)
    args = parser.parse_args()

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    detector = Detector(model_path=args.model, device=device, conf_threshold=args.conf, iou_threshold=args.iou)

    model_name = Path(args.model).stem
    output_dir = Path(args.output) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    source = args.source

    if source.startswith(('rtsp://', 'http://', 'https://')):
        print(f"Processing RTSP stream: {source}")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"rtsp_{timestamp}.mp4"
        process_video(source, detector, output_path, is_rtsp=True, show=args.show, save_mode=args.save_mode)

    elif os.path.isdir(source):
        print(f"Processing directory: {source}")
        folder_name = Path(source).name
        save_dir = output_dir / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        files = [f for f in Path(source).iterdir() if f.suffix.lower() in image_extensions]

        for file_path in tqdm.tqdm(files, desc="Processing Images"):
            img = cv2.imread(str(file_path))
            if img is None: continue
            results = detector.predict(img)
            annotated_frame = detector.draw_results(img, results)
            cv2.imwrite(str(save_dir / file_path.name), annotated_frame)

            if args.show:
                cv2.imshow("Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        if args.show:
            cv2.destroyAllWindows()
        print(f"Saved results to {save_dir}")

    elif os.path.isfile(source):
        path_obj = Path(source)
        suffix = path_obj.suffix.lower()
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        if suffix in video_extensions:
            print(f"Processing video: {source}")
            output_path = output_dir / f"out_{path_obj.name}"
            process_video(source, detector, output_path, is_rtsp=False, show=args.show, save_mode=args.save_mode)
        elif suffix in image_extensions:
            print(f"Processing image: {source}")
            img = cv2.imread(source)
            if img is not None:
                results = detector.predict(img)
                annotated_frame = detector.draw_results(img, results)
                output_path = output_dir / f"out_{path_obj.name}"
                cv2.imwrite(str(output_path), annotated_frame)
                print(f"Saved to {output_path}")
                if args.show:
                    cv2.imshow("Detection", annotated_frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            print(f"Unsupported file type: {suffix}")
    else:
        print(f"Source not found: {source}")

if __name__ == "__main__":
    main()