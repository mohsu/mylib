import datetime
import multiprocessing as mp
import threading
import queue
import time
from typing import Optional, List

import cv2
import numpy as np
from loguru import logger

now = datetime.datetime.now


class VideoYielderError(Exception):
    def __init__(self, msg):
        self.msg = msg


class VideoCaptureDaemon(threading.Thread):
    def __init__(self, URL: str, result_queue: mp.Queue):
        super().__init__()
        self.daemon = True
        self.URL = URL
        self.result_queue = result_queue

    def run(self):
        self.result_queue.put(cv2.VideoCapture(self.URL))


def get_video_capture(URL: str, timeout: Optional[float] = 5):
    res_queue = queue.Queue()
    VideoCaptureDaemon(URL, res_queue).start()
    try:
        return res_queue.get(block=True, timeout=timeout)
    except queue.Empty:
        logger.error(
            'cv2.VideoCapture: could not grab input ({}). Timeout occurred after {:.2f}s'.format(URL, timeout))
        return None


class VideoCaptureNoQueue:
    def __init__(self, URL: str, timeout: Optional[float] = 5):
        self.frame = []
        self.status = False
        self.is_stop = False
        self.count = 0

        # 攝影機連接。
        cap = None
        while cap is None:
            cap = get_video_capture(URL, timeout=timeout)
            timeout += 1
        self.capture = cap

        self.start()

    def start(self):
        # 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        threading.Thread(target=self.query_frame, daemon=True, args=()).start()

    def release(self):
        # 記得要設計停止無限迴圈的開關。
        self.is_stop = True

    def get(self, opt):
        return self.capture.get(opt)

    def read(self):
        # 當有需要影像時，再回傳最新的影像。
        return self.count, self.frame

    def query_frame(self):
        while not self.is_stop and self.capture.isOpened():
            self.status, self.frame = self.capture.read()
            self.count += 1

        self.frame = None  # if capture is stopped, put empty frame to send signal to mother process
        self.capture.release()


@logger.catch(reraise=True)
class VideoYielder:
    def __init__(self, path: str, n_frame_per_sec: Optional[int] = None, timeout: Optional[float] = 5,
                 auto_restart: bool = True):
        self.path = path
        self.n_frame_per_sec = n_frame_per_sec
        self.timeout = timeout
        self.count = 0
        self.video_retry = 0
        self.cap = None
        self.fps = 0
        self.interval = 1
        self.last_time = None
        self.auto_restart = auto_restart

    def start(self):
        self.cap = VideoCaptureNoQueue(self.path, timeout=self.timeout)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # sudo apt-get install libv4l-dev
        # self.interval = int(self.fps / self.n_frame_per_sec)
        self.interval = 1. / self.n_frame_per_sec
        self.last_time = now()
        self.count = 0
        self.video_retry = 0
        logger.debug("Stream {} is started at fps={}".format(self.path, self.fps))

    def close(self):
        logger.debug("Closing video capture.")
        self.cap.release()

    def get_image(self):
        while (now() - self.last_time).total_seconds() < self.interval:
            time.sleep(0.01)

        self.count, frame = self.cap.read()

        self.last_time = now()

        if frame is None:
            if self.auto_restart is False:
                return None
            logger.error("Cannot get image from stream, tried {}".format(self.video_retry))
            self.video_retry += 1

            if self.video_retry > 3:
                logger.error(f"max_try {self.video_retry} has reached. Restarting")

                # restart stream
                self.close()
                self.start()

                return self.get_image()
            else:
                return self.get_image()

        return frame


@logger.catch(reraise=True)
def get_video_length(video_path: str):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


@logger.catch(reraise=True)
def get_frames_from_video(video_path: str, start: float, end: float, interval: int):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start < 0:
        start = length + start
    elif start < 1:
        start = int(length * start)
    if end < 0:
        end = length + end
    elif end == 1:
        end = length
    elif end < 1:
        end = int(length * end)

    count = 1
    counts = []
    images = []
    while True:
        ret, frame = cap.read()
        if frame is not None and count <= end:
            if (interval == 1 or count % interval == 1) and start <= count <= end:
                counts.append(count)
                images.append(frame)
            count += 1
        else:
            break
    return counts, images


@logger.catch(reraise=True)
def get_frame_from_video(video_path: str, num_frame: int):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frame < 0:
        num_frame = length + num_frame
    elif num_frame < 1:
        num_frame = int(length * num_frame)
    count = 1
    while True:
        ret, image = cap.read()
        if image is not None:
            if count == num_frame:
                return image
            count += 1
        else:
            return None


@logger.catch(reraise=True)
def record_video_from_images(images: List[np.ndarray], output_path: str):
    H, W = images[0].shape[:2]
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for image in images:
        out.write(image)

    out.release()


@logger.catch(reraise=True)
def record_video_from_queue(output_path: str, queue_image: mp.Queue, fps: int = 15):
    image = None
    # get the first roulette_image
    while image is None:
        image = queue_image.get(timeout=60)
    H, W = image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    while True:
        out.write(image)
        image = queue_image.get()
        if image is None:
            break

    out.release()


@logger.catch(reraise=True)
class Recorder:
    def __init__(self, output_path: str, fps: int = 15):
        self.output_path = output_path
        self.queue_image = mp.Queue()
        self.p_record = mp.Process(target=record_video_from_queue, args=(self.output_path, self.queue_image, fps,))

    def start(self):
        self.p_record.start()
        logger.debug(f"Recorder output path {self.output_path}")

    def write(self, image):
        self.queue_image.put(image)

    def close(self):
        self.queue_image.put(None)
        self.queue_image.close()
        self.queue_image.join_thread()
