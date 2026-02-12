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
import av

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
        if self.capture.isOpened():
            self.capture.release()

class PyAVReader:
    def __init__(self, source):
        self.container = av.open(str(source))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.frames_count = self.stream.frames
        self.fps = float(self.stream.average_rate)
        self.packet_iter = self.container.demux(self.stream)
        self.frame_buffer = []

    def get_frame_count(self):
        return self.frames_count

    def get_fps(self):
        return self.fps

    def read(self):
        if self.frame_buffer:
            frame = self.frame_buffer.pop(0)
            return True, frame.to_ndarray(format='bgr24')

        try:
            for packet in self.packet_iter:
                try:
                    frames = packet.decode()
                except Exception as e:
                    # print(f"Warning: Error decoding packet: {e}")
                    continue

                if frames:
                    first_frame = frames.pop(0)
                    self.frame_buffer.extend(frames)
                    return True, first_frame.to_ndarray(format='bgr24')

            return False, None
        except Exception as e:
            # print(f"Error reading stream: {e}")
            return False, None

    def grab(self):
        ret, _ = self.read()
        return ret

    def release(self):
        self.container.close()

class Detector:
    def __init__(self, model_path='yolov8n.pt', device='cpu', conf_threshold=0.25, iou_threshold=0.45):
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model.to(self.device)

    def predict(self, image):
        return self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)

class CropConfig:
    def __init__(self, target_classes=None, min_size=0, use_tracking=False, similarity_thresh=0.0):
        self.target_classes = target_classes if target_classes else []
        self.min_size = min_size
        self.use_tracking = use_tracking
        self.similarity_thresh = similarity_thresh

class MotionDetector:
    def __init__(self, resize_width=480, blur_ksize=(21, 21), delta_thresh=25, min_area_percent=0.005):
        self.resize_width = resize_width
        self.blur_ksize = blur_ksize
        self.delta_thresh = delta_thresh
        self.min_area_percent = min_area_percent
        self.prev_gray = None
        self.min_area = 0

    def is_motion(self, frame):
        h, w = frame.shape[:2]
        r = self.resize_width / float(w)
        dim = (self.resize_width, int(h * r))

        # 1. Resize
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # 2. Convert to Gray
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 3. Gaussian Blur (Heavy blur to remove H.264 noise)
        gray = cv2.GaussianBlur(gray, self.blur_ksize, 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.min_area = (dim[0] * dim[1]) * self.min_area_percent
            return True # Always treat first frame as motion

        # 4. Frame Difference
        frame_delta = cv2.absdiff(self.prev_gray, gray)

        # 5. Threshold
        thresh = cv2.threshold(frame_delta, self.delta_thresh, 255, cv2.THRESH_BINARY)[1]

        # 6. Dilate
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 7. Check contours area
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        has_motion = False
        for c in contours:
            if cv2.contourArea(c) > self.min_area:
                has_motion = True
                break

        self.prev_gray = gray

        return has_motion

def compare_histograms(img1, img2):
    try:
        if img1.size == 0 or img2.size == 0: return 0.0

        h1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        h2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)

        return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    except:
        return 0.0

def process_video_export(source, detector, output_dir, is_rtsp=False, show=False, crop_config=None, target_fps=-1, motion_config=None):
    if crop_config is None:
        crop_config = CropConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_interval = 1

    # Motion Detector Setup
    motion_detector = None
    if motion_config:
        motion_detector = MotionDetector(
            resize_width=motion_config['resize_width'],
            blur_ksize=motion_config['blur_ksize'],
            delta_thresh=motion_config['delta_thresh'],
            min_area_percent=motion_config['min_area_percent']
        )

    if is_rtsp:
        stream = RTSPStream(source)
        time.sleep(1.0) # Warmup
        status, frame = stream.read()
        total_frames = 0
    else:
        # Replaced cv2.VideoCapture with PyAVReader
        try:
            cap = PyAVReader(source)
            total_frames = cap.get_frame_count()
            status, frame = cap.read() # Read first frame for valid check

            if target_fps > 0:
                vid_fps = cap.get_fps()
                if vid_fps > target_fps:
                    frame_interval = max(1, int(round(vid_fps / target_fps)))
                    print(f"Original FPS: {vid_fps:.2f}, Target FPS: {target_fps}. Processing every {frame_interval} frames.")
        except Exception as e:
            print(f"Error opening source with PyAV: {source}, {e}")
            return

    if not status or frame is None:
        print(f"Error: Source empty or failed to open.")
        return

    pbar = None
    if not is_rtsp and total_frames > 0:
        pbar = tqdm.tqdm(total=total_frames, desc="Processing Video")

    track_history = {}
    frame_idx = 0

    consecutive_static_count = 0

    try:
        while True:
            # 1. Read Frame
            if is_rtsp:
                status, frame = stream.read()
                if not status:
                    time.sleep(0.01)
                    continue
            else:
                # Video file skipping logic
                # We use cap.grab() which in PyAVReader just effectively skips a frame in the iterator
                if frame_interval > 1:
                    if (frame_idx) % frame_interval != 0:
                         ret = cap.grab()
                         if not ret: break
                         frame_idx += 1
                         if pbar: pbar.update(1)
                         continue

                status, frame = cap.read()
                if not status:
                    break

            frame_idx += 1

            # 2. Motion Detection (For both RTSP and Video)
            if motion_detector:
                # Perform check
                if not motion_detector.is_motion(frame):
                    # Static detected
                    consecutive_static_count += 1

                    if is_rtsp:
                        # Sleep to save CPU on static RTSP
                        time.sleep(0.02)

                    if show:
                        # Optional: Draw specific label for static
                        vis_frame = frame.copy()
                        cv2.putText(vis_frame, "Static Scenery - Skipped", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.imshow("Crop Export", vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'): break

                    if pbar: pbar.update(1) # Update pbar for skipped frames
                    continue # Skip YOLO
                else:
                    consecutive_static_count = 0

            # 3. YOLO Predict/Track
            if crop_config.use_tracking:
                results = detector.model.track(frame, conf=detector.conf_threshold, iou=detector.iou_threshold, persist=True, verbose=False)
            else:
                results = detector.predict(frame)

            annotated_frame = frame.copy()

            # 4. Process Results
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])

                    if crop_config.target_classes and cls_id not in crop_config.target_classes: continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    w, h = x2 - x1, y2 - y1

                    if w < crop_config.min_size or h < crop_config.min_size: continue

                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img.size == 0: continue

                    track_id = int(box.id[0]) if box.id is not None else -1

                    save_this = True
                    if crop_config.use_tracking and track_id != -1 and crop_config.similarity_thresh > 0:
                        if track_id in track_history:
                            sim = compare_histograms(crop_img, track_history[track_id])
                            if sim >= crop_config.similarity_thresh:
                                save_this = False

                        if save_this:
                            track_history[track_id] = crop_img.copy()

                    if save_this:
                        obj_idx = track_id if track_id != -1 else i
                        filename = f"{frame_idx}_{obj_idx}_{cls_id}.jpg"
                        cv2.imwrite(str(output_dir / filename), crop_img)

                        # Vis
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{detector.model.names[cls_id]}"
                        if track_id != -1: label += f" ID:{track_id}"
                        cv2.putText(annotated_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if show:
                cv2.imshow("Crop Export", annotated_frame)
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
            if 'cap' in locals(): cap.release()
            if pbar: pbar.close()

        if show:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Input source: image, video, directory, or RTSP URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--conf', type=float, default=0.45, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--device', type=str, default='', help='Device (cpu, cuda, 0, 1...)')
    parser.add_argument('--output', type=str, default='output_crops', help='Output directory')
    parser.add_argument('--show', action='store_true', help='Show real-time detection results')

    # Crop specific args
    parser.add_argument('--crop_classes', nargs='+', default=[], help='List of class names or IDs to crop (empty=all)')
    parser.add_argument('--min_size', type=int, default=0, help='Minimum width/height for cropped images')
    parser.add_argument('--use_tracking', action='store_true', help='Enable tracking for video/stream sources')
    parser.add_argument('--similarity_thresh', type=float, default=0.0, help='Similarity threshold (0.0-1.0) to skip saving similar objects (requires tracking)')
    parser.add_argument('--target_fps', type=int, default=-1, help='Target FPS for processing (skip frames). Default -1 (original FPS).')

    # Motion Detection args
    parser.add_argument('--disable_motion_filter', action='store_true', help='Disable motion detection filter for RTSP/Video')
    parser.add_argument('--motion_resize_w', type=int, default=480, help='Resize width for motion detection analysis')
    parser.add_argument('--motion_blur_k', type=int, default=21, help='Gaussian Blur kernel size (odd number)')
    parser.add_argument('--motion_thresh', type=int, default=25, help='Delta threshold for motion detection (0-255)')
    parser.add_argument('--motion_area', type=float, default=0.001, help='Minimum motion area percentage (0.0 - 1.0)')

    # Ignored args if present (compat)
    parser.add_argument('--save_mode', type=str, default='all', help='Ignored in crop mode')

    args = parser.parse_args()

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    detector = Detector(model_path=args.model, device=device, conf_threshold=args.conf, iou_threshold=args.iou)

    # Resolve classes
    target_classes = []
    if args.crop_classes:
        for c in args.crop_classes:
            if c.isdigit():
                target_classes.append(int(c))
            else:
                found = False
                for k, v in detector.model.names.items():
                    if v.lower() == c.lower():
                        target_classes.append(k)
                        found = True
                        break
                if not found:
                    print(f"Warning: Class '{c}' not found in model.")

    crop_config = CropConfig(
        target_classes=target_classes,
        min_size=args.min_size,
        use_tracking=args.use_tracking,
        similarity_thresh=args.similarity_thresh
    )

    # Motion Config
    motion_config = None
    source = args.source
    is_rtsp = source.startswith(('rtsp://', 'http://', 'https://'))

    if not args.disable_motion_filter:
        if args.motion_blur_k % 2 == 0:
            print("Warning: Motion blur kernel size must be odd. Adjusting to +1.")
            args.motion_blur_k += 1

        motion_config = {
            'resize_width': args.motion_resize_w,
            'blur_ksize': (args.motion_blur_k, args.motion_blur_k),
            'delta_thresh': args.motion_thresh,
            'min_area_percent': args.motion_area
        }
        print(f"Motion Filter Enabled: Resize={args.motion_resize_w}, Blur={args.motion_blur_k}, Thresh={args.motion_thresh}, MinArea={args.motion_area}")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if is_rtsp:
        # Stream
        folder_name = f"stream_{time.strftime('%Y%m%d_%H%M%S')}"
        output_dir = output_root / folder_name
        print(f"Processing Stream: {source} -> {output_dir}")
        process_video_export(source, detector, output_dir, is_rtsp=True, show=args.show, crop_config=crop_config, target_fps=args.target_fps, motion_config=motion_config)

    elif os.path.isdir(source):
        print(f"Processing Directory: {source}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        files = [f for f in Path(source).iterdir() if f.suffix.lower() in image_extensions]

        for file_path in tqdm.tqdm(files, desc="Processing Images"):
            img = cv2.imread(str(file_path))
            if img is None: continue

            results = detector.predict(img)

            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    if target_classes and cls_id not in target_classes: continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    w, h = x2 - x1, y2 - y1
                    if w < args.min_size or h < args.min_size: continue

                    crop_img = img[y1:y2, x1:x2]
                    if crop_img.size == 0: continue

                    filename = f"{file_path.stem}_{i}_{cls_id}.jpg"
                    cv2.imwrite(str(output_root / filename), crop_img)

    elif os.path.isfile(source):
        path_obj = Path(source)
        suffix = path_obj.suffix.lower()
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.ts'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        if suffix in video_extensions:
            folder_name = path_obj.stem
            output_dir = output_root / folder_name
            print(f"Processing Video: {source} -> {output_dir}")
            process_video_export(source, detector, output_dir, is_rtsp=False, show=args.show, crop_config=crop_config, target_fps=args.target_fps, motion_config=motion_config)

        elif suffix in image_extensions:
            print(f"Processing Image: {source}")
            img = cv2.imread(source)
            if img is None:
                print("Failed to load image")
            else:
                results = detector.predict(img)
                for result in results:
                    boxes = result.boxes
                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        if target_classes and cls_id not in target_classes: continue

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        w, h = x2 - x1, y2 - y1
                        if w < args.min_size or h < args.min_size: continue

                        crop_img = img[y1:y2, x1:x2]
                        if crop_img.size == 0: continue

                        filename = f"{path_obj.stem}_{i}_{cls_id}.jpg"
                        cv2.imwrite(str(output_root / filename), crop_img)
                        print(f"Saved {filename}")
        else:
            print(f"Unsupported file type: {suffix}")
    else:
        print(f"Source not found: {source}")
