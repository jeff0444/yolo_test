import av
import queue
import time
import numpy as np
from pathlib import Path
from typing import Literal

class EstimatedFPS:
    def __init__(self, timePrecision: Literal['s', 'ms', 'us', 'ns'] = 'ms'):
        self.__Interval_ts = queue.Queue(maxsize=32)
        self.__last_frame_ts = 0
        self.__estimated_fps: float = 0.0
        timePrecisionMap = dict(s=1.0, ms=1000.0, us=1000000.0, ns=1000000000.0)
        self.__one_second_number = timePrecisionMap.get(timePrecision, timePrecisionMap['ms'])

    @property
    def get_fps(self) -> float:
        return self.__estimated_fps

    def update(self, current_frame_ts):
        if self.__last_frame_ts > 0:
            between_frames_num = current_frame_ts - self.__last_frame_ts
            if between_frames_num <= 0:
                pass # ignore invalid intervals
            else:
                if self.__Interval_ts.full():
                    self.__Interval_ts.get()
                self.__Interval_ts.put(between_frames_num)

                intervals = list(self.__Interval_ts.queue)
                if len(intervals) > 0:
                    avg_time = sum(intervals) / len(intervals)
                    self.__estimated_fps = self.__one_second_number / avg_time if avg_time > 0 else self.__estimated_fps
        self.__last_frame_ts = current_frame_ts

class SaveStream_ffmpeg:
    def __init__(self, filename, fps=-1, width=640, height=480, crf=35):
        self.__filename = filename
        self.__fps = fps
        self.__width = width
        self.__height = height
        self.__crf = crf

        self.__container = None
        self.__v_stream = None
        self.frameBuffer = []
        self.estimated_fps = EstimatedFPS(timePrecision='ms')

    def __createStream(self, fps):
        Path(self.__filename).parent.mkdir(parents=True, exist_ok=True)
        print(f"Creating output video file: {self.__filename} with FPS: {fps:.2f}")
        try:
            self.__container = av.open(str(self.__filename), mode='w')
            self.__v_stream = self.__container.add_stream('h264', rate=int(round(fps)))
            self.__v_stream.width = self.__width
            self.__v_stream.height = self.__height
            self.__v_stream.pix_fmt = 'yuv420p'
            self.__v_stream.options = {'crf': str(self.__crf)}
        except Exception as e:
            print(f"Failed to create stream: {e}")
            self.__container = None

    def update_fps(self, timestamp):
        self.estimated_fps.update(timestamp)

    def write(self, frame, timestamp=None, update_fps=True):
        if timestamp is None:
            timestamp = time.time() * 1000

        # Update FPS estimator
        if update_fps:
            self.estimated_fps.update(timestamp)

        # Add to buffer
        self.frameBuffer.append(frame)

        # Check if we should initialize stream
        if self.__container is None:
            if self.__fps > 0:
                self.__createStream(self.__fps)
            elif len(self.frameBuffer) >= 5:
                estimated = self.estimated_fps.get_fps
                if estimated > 0:
                    self.__createStream(estimated)
                # If still 0, we keep buffering until we get a valid FPS or user stops?
                # Or we can set a fallback like 30 if buffer gets too large?
                elif len(self.frameBuffer) > 60:
                    print("Could not estimate FPS, defaulting to 30")
                    self.__createStream(30)

        # Write frames if stream is ready
        if self.__container is not None:
            while self.frameBuffer:
                img_bgr = self.frameBuffer.pop(0)
                # Convert BGR (OpenCV) to RGB or YUV for PyAV?
                # av.VideoFrame.from_ndarray expects format. 'bgr24' is supported.
                frame_av = av.VideoFrame.from_ndarray(img_bgr, format='bgr24')

                # We don't manually set PTS here, relying on rate and sequential arrival
                try:
                    for packet in self.__v_stream.encode(frame_av):
                        self.__container.mux(packet)
                except Exception as e:
                    print(f"Error encoding frame: {e}")

    def close(self):
        if self.__container is not None:
             # Flush buffer if stream was open but frames remained (shouldn't happen with current logic unless close called mid-buffer processing?)
             # Actually if FPS never detected, we might lose buffer.
             # On close, if buffer exists and no stream, we should probably force create stream.
            if self.frameBuffer and self.__container is None:
                fps = self.estimated_fps.get_fps
                if fps <= 0: fps = 30
                self.__createStream(fps)

            # Now flush buffer
            while self.frameBuffer:
                img_bgr = self.frameBuffer.pop(0)
                frame_av = av.VideoFrame.from_ndarray(img_bgr, format='bgr24')
                try:
                    for packet in self.__v_stream.encode(frame_av):
                        self.__container.mux(packet)
                except Exception as e:
                     print(e)

            # Flush encoder
            if self.__v_stream:
                try:
                    for packet in self.__v_stream.encode():
                        self.__container.mux(packet)
                except Exception as e:
                    print(e)

            self.__container.close()
            print(f"Video saved to {self.__filename}")
        else:
             # Case where stream never started (e.g. very few frames and FPS <=0)
             if self.frameBuffer:
                 # Fallback
                 self.__createStream(30)
                 self.close() # Recursively save
