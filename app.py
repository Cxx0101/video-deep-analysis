from __future__ import annotations

import json
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import re
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import mediapipe as mp
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse


# The web service keeps all mutable files inside the project directory.
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR
DATA_DIR = STORAGE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
URL_DOWNLOAD_DIR = DATA_DIR / "url_downloads"
COOKIE_DIR = DATA_DIR / "download_cookies"
COOKIE_STORE = DATA_DIR / "cookies.txt"
MODEL_DIR = STORAGE_DIR / "models"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_COOKIE_BYTES = 5 * 1024 * 1024
for directory in (UPLOAD_DIR, RESULT_DIR, URL_DOWNLOAD_DIR, COOKIE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def configure_mediapipe_resource_path() -> None:
    """Make MediaPipe's native resource loader work from Unicode Windows paths.

    Some MediaPipe Windows builds pass resource paths through a non-Unicode
    native layer.  A user can legitimately keep this project in a directory
    with Chinese characters, so expose site-packages through an ASCII junction
    when the installed path is not ASCII.  The junction only contains a link;
    model data remains in the project directory.
    """
    if os.name != "nt":
        return
    try:
        import mediapipe.python.solution_base as solution_base

        site_packages = Path(mp.__file__).resolve().parent.parent
        if site_packages.as_posix().isascii():
            return
        cache_root = Path(os.environ.get("LOCALAPPDATA", "")) / "VideoDeepAnalysis" / "mediapipe"
        if not cache_root.as_posix().isascii():
            return
        alias = cache_root / uuid.uuid5(uuid.NAMESPACE_URL, str(site_packages)).hex
        alias.parent.mkdir(parents=True, exist_ok=True)
        if not alias.exists():
            created = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(site_packages)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if created.returncode:
                return
        patched_file = alias / "mediapipe" / "python" / "solution_base.py"
        if patched_file.is_file():
            solution_base.__file__ = str(patched_file)
    except (OSError, subprocess.SubprocessError):
        # MediaPipe will retain its normal behavior on systems where a junction
        # is unavailable; this should not prevent the web service from starting.
        return


configure_mediapipe_resource_path()


def native_readable_path(path: Path) -> Path:
    """Return an ASCII Windows junction path when a native library needs one."""
    if os.name != "nt" or path.as_posix().isascii():
        return path
    try:
        cache_root = Path(os.environ.get("LOCALAPPDATA", "")) / "VideoDeepAnalysis" / "path-links"
        if not cache_root.as_posix().isascii():
            return path
        source_dir = path.resolve().parent
        alias = cache_root / uuid.uuid5(uuid.NAMESPACE_URL, str(source_dir)).hex
        alias.parent.mkdir(parents=True, exist_ok=True)
        if not alias.exists():
            created = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(source_dir)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if created.returncode:
                return path
        candidate = alias / path.name
        return candidate if candidate.is_file() else path
    except (OSError, subprocess.SubprocessError):
        return path


# Keep all downloaded model files beside the project.

MODES = {
    "depth": "灰度深度图",
    "pose": "人体姿态骨架叠加",
    "depth_pose": "深度图 + 人体姿态骨架",
    "face": "面部 478 点网格",
    "all": "深度图 + 人体姿态 + 面部网格",
}
PORTRAIT_MODES = {
    "blur": "背景虚化",
    "white": "白色背景",
    "black": "黑色背景",
    "mask": "人像蒙版",
    "transparent": "透明背景（WebM）",
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}

jobs: dict[str, dict[str, Any]] = {}
# A source is intentionally separate from a processing job.  A local upload or
# downloaded URL can therefore be previewed once and passed to any current or
# future video tool without uploading/downloading it again.
sources: dict[str, dict[str, Any]] = {}
models_lock = threading.Lock()
processing_lock = threading.Lock()


def ffmpeg_executable() -> str:
    """Use a system FFmpeg when present, otherwise imageio-ffmpeg's binary."""
    system_binary = shutil.which("ffmpeg")
    if system_binary:
        return system_binary
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("未找到 FFmpeg，无法生成网页兼容的 H.264 MP4。请重新安装项目依赖。") from exc


def source_rotation_degrees(video_path: Path) -> int:
    """Return a video's display rotation without relying on OpenCV alone."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate:stream_side_data=rotation",
        "-of", "json", str(video_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        stream = json.loads(completed.stdout).get("streams", [{}])[0]
        side_data = stream.get("side_data_list", [])
        raw_rotation = next((item.get("rotation") for item in side_data if "rotation" in item), None)
        if raw_rotation is None:
            raw_rotation = stream.get("tags", {}).get("rotate")
        return int(round(float(raw_rotation))) % 360 if raw_rotation is not None else 0
    except (IndexError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return 0


def encode_browser_mp4(silent_video: Path, source_video: Path, output_path: Path) -> None:
    """Encode H.264/AAC MP4, preserve source audio, and move metadata up front."""
    command = [
        ffmpeg_executable(), "-y",
        "-i", str(silent_video),
        "-i", str(source_video),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-map_metadata", "-1",
        "-metadata:s:v:0", "rotate=0",
        "-movflags", "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "未知 FFmpeg 错误"
        raise RuntimeError(f"MP4 转码失败：{detail}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("MP4 转码未生成有效输出文件。")


def source_rotation_filter(video_path: Path) -> str | None:
    """Return an FFmpeg filter that turns display rotation into real pixels."""
    rotation = source_rotation_degrees(video_path)
    if rotation == 90:
        return "transpose=2"  # counter-clockwise, matching the analysis renderer
    if rotation == 180:
        return "hflip,vflip"
    if rotation == 270:
        return "transpose=1"
    return None


def separate_audio_video(source_video: Path, video_output: Path, audio_output: Path) -> None:
    """Export a browser-ready silent MP4 and the source's first audio track."""
    video_command = [
        ffmpeg_executable(), "-y", "-noautorotate", "-i", str(source_video),
        "-map", "0:v:0", "-an",
    ]
    rotation_filter = source_rotation_filter(source_video)
    if rotation_filter:
        video_command.extend(["-vf", rotation_filter])
    video_command.extend([
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-map_metadata", "-1",
        "-metadata:s:v:0", "rotate=0", "-movflags", "+faststart", str(video_output),
    ])
    audio_command = [
        ffmpeg_executable(), "-y", "-i", str(source_video),
        "-map", "0:a:0", "-vn", "-c:a", "libmp3lame", "-q:a", "2", str(audio_output),
    ]
    try:
        for command, label in ((video_command, "无声视频"), (audio_command, "音频")):
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if completed.returncode != 0:
                detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "未知 FFmpeg 错误"
                if label == "音频":
                    raise RuntimeError(f"音频导出失败：该视频可能不含音轨，或音频格式不受支持。{detail}")
                raise RuntimeError(f"{label}导出失败：{detail}")
        if not video_output.is_file() or video_output.stat().st_size == 0:
            raise RuntimeError("无声视频导出未生成有效文件。")
        if not audio_output.is_file() or audio_output.stat().st_size == 0:
            raise RuntimeError("音频导出未生成有效文件。")
    except Exception:
        video_output.unlink(missing_ok=True)
        audio_output.unlink(missing_ok=True)
        raise


class VideoAnalyzer:
    """Lazily loads CPU depth and landmark models once per process."""

    def __init__(self) -> None:
        self.depth_net: cv2.dnn.Net | None = None
        self.depth_session: Any | None = None
        self.depth_input_name: str | None = None
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.drawer = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles

    @property
    def device_name(self) -> str:
        return "CPU"

    def load_depth_model(self) -> None:
        if self.depth_net is not None or self.depth_session is not None:
            return
        # MiDaS Small ONNX runs through OpenCV DNN on CPU. This avoids installing
        # the much larger PyTorch/CUDA runtime while preserving depth analysis.
        with models_lock:
            if self.depth_net is not None or self.depth_session is not None:
                return
            weights = MODEL_DIR / "midas_v21_small_256.onnx"
            min_model_bytes = 50 * 1024 * 1024
            if not weights.is_file() or weights.stat().st_size < min_model_bytes:
                from urllib.request import urlretrieve

                partial = weights.with_suffix(".onnx.part")
                partial.unlink(missing_ok=True)
                try:
                    urlretrieve("https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx", partial)
                    if partial.stat().st_size < min_model_bytes:
                        raise RuntimeError("下载的深度模型不完整")
                    partial.replace(weights)
                finally:
                    partial.unlink(missing_ok=True)
            model_path = native_readable_path(weights)
            try:
                net = cv2.dnn.readNetFromONNX(str(model_path))
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.depth_net = net
                return
            except cv2.error as opencv_error:
                # Some macOS OpenCV wheels reject this legacy ONNX graph.  Use
                # the CPU-only ONNX Runtime fallback there without reintroducing
                # PyTorch or CUDA dependencies on Windows.
                if sys.platform != "darwin":
                    raise RuntimeError(f"深度模型加载失败：{opencv_error}") from opencv_error
                try:
                    import onnxruntime as ort

                    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                    self.depth_session = session
                    self.depth_input_name = session.get_inputs()[0].name
                    return
                except Exception as runtime_error:
                    raise RuntimeError(
                        f"深度模型加载失败。OpenCV：{opencv_error}；ONNX Runtime：{runtime_error}"
                    ) from runtime_error

    def depth_frame(self, rgb: np.ndarray) -> np.ndarray:
        self.load_depth_model()
        assert self.depth_net is not None or self.depth_session is not None
        # These values are the official MiDaS v2.1 Small ONNX preprocessing.
        blob = cv2.dnn.blobFromImage(
            rgb, 1 / 255.0, (256, 256), (123.675, 116.28, 103.53), swapRB=True, crop=False
        )
        if self.depth_net is not None:
            self.depth_net.setInput(blob)
            prediction = self.depth_net.forward()[0]
        else:
            assert self.depth_session is not None and self.depth_input_name is not None
            prediction = self.depth_session.run(None, {self.depth_input_name: blob})[0][0]
        values = cv2.resize(prediction, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        # Normalize each frame for readable relative depth. Brighter = nearer.
        normalized = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    def render(self, input_path: Path, output_path: Path, mode: str, progress: dict[str, Any]) -> None:
        include_depth = mode in {"depth", "depth_pose", "all"}
        include_pose = mode in {"pose", "depth_pose", "all"}
        include_face = mode in {"face", "all"}
        if include_depth:
            self.load_depth_model()

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("无法读取该视频。请上传常见的 MP4、MOV、AVI 或 WebM 视频。")
        orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        if orientation_auto is not None:
            capture.set(orientation_auto, 0)
        rotation = source_rotation_degrees(input_path)
        if not rotation:
            orientation_meta = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
            if orientation_meta is not None:
                rotation = int(round(capture.get(orientation_meta))) % 360
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps > 240:
            fps = 30.0
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if rotation in {90, 270}:
            width, height = height, width
        if width < 2 or height < 2:
            raise ValueError("视频尺寸无效。")
        frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        silent_output = output_path.with_name(f"{output_path.stem}.silent.mp4")
        writer = cv2.VideoWriter(str(silent_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("无法创建 MP4 输出文件。")

        pose_context = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=1, enable_segmentation=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) if include_pose else None
        face_context = self.mp_face.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) if include_face else None
        try:
            index = 0
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                if rotation == 90:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == 180:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_180)
                elif rotation == 270:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                canvas = self.depth_frame(rgb) if include_depth else bgr.copy()
                if pose_context is not None:
                    pose_result = pose_context.process(rgb)
                    if pose_result.pose_landmarks:
                        self.drawer.draw_landmarks(
                            canvas, pose_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.styles.get_default_pose_landmarks_style(),
                        )
                if face_context is not None:
                    face_result = face_context.process(rgb)
                    if face_result.multi_face_landmarks:
                        for landmarks in face_result.multi_face_landmarks:
                            self.drawer.draw_landmarks(
                                canvas, landmarks, self.mp_face.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.styles.get_default_face_mesh_tesselation_style(),
                            )
                            self.drawer.draw_landmarks(
                                canvas, landmarks, self.mp_face.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.styles.get_default_face_mesh_contours_style(),
                            )
                writer.write(canvas)
                index += 1
                progress["progress"] = round(index / frame_count * 100, 1)
                progress["message"] = f"正在处理第 {index}/{frame_count} 帧"
        finally:
            capture.release()
            writer.release()
            if pose_context:
                pose_context.close()
            if face_context:
                face_context.close()
        try:
            progress.update(progress=98, message="正在保留原始音频并转码为网页兼容 MP4…")
            encode_browser_mp4(silent_output, input_path, output_path)
        finally:
            silent_output.unlink(missing_ok=True)

    def render_transparent_portrait(self, input_path: Path, output_path: Path, progress: dict[str, Any]) -> None:
        """Write a VP9 WebM with an alpha channel and the original audio."""
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("无法读取该视频。请上传常见的视频格式后重试。")
        orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        if orientation_auto is not None:
            capture.set(orientation_auto, 0)
        rotation = source_rotation_degrees(input_path)
        if not rotation:
            orientation_meta = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
            if orientation_meta is not None:
                rotation = int(round(capture.get(orientation_meta))) % 360
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps > 240:
            fps = 30.0
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if rotation in {90, 270}:
            width, height = height, width
        if width < 2 or height < 2:
            capture.release()
            raise ValueError("视频尺寸无效。")
        frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        command = [
            ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra",
            "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "-",
            "-i", str(input_path), "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "32",
            "-auto-alt-ref", "0", "-c:a", "libopus", "-b:a", "128k", "-shortest",
            "-map_metadata", "-1", "-metadata:s:v:0", "alpha_mode=1", str(output_path),
        ]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
        try:
            assert encoder.stdin is not None
            index = 0
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                if rotation == 90:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == 180:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_180)
                elif rotation == 270:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = segmenter.process(rgb)
                mask = result.segmentation_mask
                if mask is None:
                    raise RuntimeError("未能生成人像蒙版。")
                alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)
                alpha = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
                bgra = np.dstack((bgr, alpha))
                encoder.stdin.write(np.ascontiguousarray(bgra).tobytes())
                index += 1
                progress["progress"] = round(index / frame_count * 100, 1)
                progress["message"] = f"正在生成透明人像视频：{index}/{frame_count} 帧"
        except Exception:
            if encoder.stdin is not None:
                encoder.stdin.close()
            encoder.wait()
            output_path.unlink(missing_ok=True)
            raise
        finally:
            capture.release()
            segmenter.close()
        assert encoder.stdin is not None and encoder.stderr is not None
        encoder.stdin.close()
        error_output = encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.wait() != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
            detail = error_output.strip().splitlines()[-1] if error_output.strip() else "未知 FFmpeg 错误"
            raise RuntimeError(f"透明 WebM 导出失败：{detail}")

    def render_portrait(
        self, input_path: Path, output_path: Path, portrait_mode: str, progress: dict[str, Any]
    ) -> None:
        """Separate a person from the background and preserve the source audio."""
        if portrait_mode not in PORTRAIT_MODES:
            raise ValueError("未知的人像与背景分离模式")
        if portrait_mode == "transparent":
            self.render_transparent_portrait(input_path, output_path, progress)
            return
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("无法读取该视频。请上传常见的视频格式后重试。")
        orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        if orientation_auto is not None:
            capture.set(orientation_auto, 0)
        rotation = source_rotation_degrees(input_path)
        if not rotation:
            orientation_meta = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
            if orientation_meta is not None:
                rotation = int(round(capture.get(orientation_meta))) % 360
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps > 240:
            fps = 30.0
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if rotation in {90, 270}:
            width, height = height, width
        if width < 2 or height < 2:
            capture.release()
            raise ValueError("视频尺寸无效。")
        frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        silent_output = output_path.with_name(f"{output_path.stem}.silent.mp4")
        writer = cv2.VideoWriter(str(silent_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("无法创建人像分离输出文件。")

        segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
        try:
            index = 0
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                if rotation == 90:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == 180:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_180)
                elif rotation == 270:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = segmenter.process(rgb)
                mask = result.segmentation_mask
                if mask is None:
                    raise RuntimeError("未能生成人像蒙版。")
                alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)
                alpha = np.clip(alpha, 0.0, 1.0)[..., None]
                if portrait_mode == "mask":
                    values = (alpha[..., 0] >= 0.5).astype(np.uint8) * 255
                    canvas = cv2.cvtColor(values, cv2.COLOR_GRAY2BGR)
                else:
                    if portrait_mode == "blur":
                        background = cv2.GaussianBlur(bgr, (0, 0), 25)
                    elif portrait_mode == "white":
                        background = np.full_like(bgr, 255)
                    else:
                        background = np.zeros_like(bgr)
                    canvas = (bgr.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
                writer.write(canvas)
                index += 1
                progress["progress"] = round(index / frame_count * 100, 1)
                progress["message"] = f"正在分离人像与背景：{index}/{frame_count} 帧"
        finally:
            capture.release()
            writer.release()
            segmenter.close()
        try:
            progress.update(progress=98, message="正在保留原始音频并转码为网页兼容 MP4…")
            encode_browser_mp4(silent_output, input_path, output_path)
        finally:
            silent_output.unlink(missing_ok=True)


analyzer = VideoAnalyzer()
app = FastAPI(title="视频深度分析工具")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": analyzer.device_name}


@app.get("/api/cookies/status")
def cookie_status() -> dict[str, Any]:
    if not COOKIE_STORE.is_file():
        return {"saved": False}
    return {
        "saved": True,
        "updated_at": int(COOKIE_STORE.stat().st_mtime),
    }


@app.delete("/api/cookies")
def clear_saved_cookies() -> dict[str, bool]:
    COOKIE_STORE.unlink(missing_ok=True)
    return {"cleared": True}


@app.post("/api/cookies")
async def save_cookies(cookie_file: UploadFile = File(...)) -> dict[str, bool]:
    await update_cookie_store(cookie_file)
    return {"saved": True}


def register_source(
    source_id: str,
    path: Path,
    name: str,
    origin: str,
    source_dir: Path | None = None,
) -> None:
    sources[source_id] = {
        "path": str(path),
        "name": name,
        "origin": origin,
        "source_dir": str(source_dir) if source_dir else None,
    }


async def save_uploaded_source(video: UploadFile) -> tuple[str, Path, str]:
    """Save one uploaded video so it can be reused by multiple tool jobs."""
    original_name = Path(video.filename or "video.mp4").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        await video.close()
        raise HTTPException(400, "请上传 MP4、MOV、AVI、MKV、WebM 或 M4V 视频")
    source_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{source_id}{suffix}"
    size = 0
    try:
        with input_path.open("wb") as destination:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "视频超过 500 MB 限制")
                destination.write(chunk)
    except Exception:
        input_path.unlink(missing_ok=True)
        raise
    finally:
        await video.close()
    register_source(source_id, input_path, original_name, "upload")
    return source_id, input_path, original_name


@app.post("/api/sources/upload")
async def upload_source(video: UploadFile = File(...)) -> dict[str, str]:
    """Create a reusable source from a local upload before selecting a tool."""
    source_id, _input_path, original_name = await save_uploaded_source(video)
    return {
        "source_id": source_id,
        "name": original_name,
        "preview_url": f"/api/sources/{source_id}/preview",
    }


def get_source_path(source_id: str) -> tuple[dict[str, Any], Path]:
    source = sources.get(source_id)
    path = Path(str(source.get("path", ""))) if source else None
    if not source or not path.is_file():
        raise HTTPException(404, "视频来源不存在，或服务已重启")
    return source, path


@app.get("/api/sources/{source_id}/preview")
def preview_source(source_id: str) -> FileResponse:
    _source, path = get_source_path(source_id)
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/sources/{source_id}/jobs")
async def create_source_job(
    source_id: str,
    background_tasks: BackgroundTasks,
    tool: str = Form(...),
    mode: str | None = Form(None),
    portrait_mode: str | None = Form(None),
) -> dict[str, str]:
    """Run a selected tool against an existing uploaded/downloaded source."""
    _source, input_path = get_source_path(source_id)
    if tool not in {"analysis", "separate", "portrait"}:
        raise HTTPException(400, "未知的视频工具")
    if tool == "analysis" and mode not in MODES:
        raise HTTPException(400, "请选择有效的分析模式")
    if tool == "portrait" and portrait_mode not in PORTRAIT_MODES:
        raise HTTPException(400, "请选择有效的人像与背景分离模式")

    job_id = uuid.uuid4().hex
    if tool == "analysis":
        output_path = RESULT_DIR / f"{job_id}.mp4"
        jobs[job_id] = {
            "status": "queued", "progress": 0, "message": "处理任务已排队",
            "tool": tool, "mode": mode, "source_id": source_id, "output": str(output_path),
        }
        background_tasks.add_task(process_source_analysis_job, job_id, input_path, output_path, str(mode))
    elif tool == "separate":
        video_output = RESULT_DIR / f"{job_id}-silent.mp4"
        audio_output = RESULT_DIR / f"{job_id}-audio.mp3"
        jobs[job_id] = {
            "status": "queued", "progress": 0, "message": "音视频分离任务已排队",
            "tool": tool, "source_id": source_id,
            "outputs": {"video": str(video_output), "audio": str(audio_output)},
        }
        background_tasks.add_task(process_source_separation_job, job_id, input_path, video_output, audio_output)
    else:
        output_suffix = ".webm" if portrait_mode == "transparent" else ".mp4"
        output_path = RESULT_DIR / f"{job_id}-portrait{output_suffix}"
        jobs[job_id] = {
            "status": "queued", "progress": 0, "message": "人像与背景分离任务已排队",
            "tool": tool, "portrait_mode": portrait_mode, "source_id": source_id, "output": str(output_path),
        }
        background_tasks.add_task(process_source_portrait_job, job_id, input_path, output_path, str(portrait_mode))
    return {"job_id": job_id}


def process_source_analysis_job(job_id: str, input_path: Path, output_path: Path, mode: str) -> None:
    job = jobs[job_id]
    try:
        with processing_lock:
            job.update(status="processing", message="正在加载分析模型…")
            analyzer.render(input_path, output_path, mode, job)
        job.update(status="completed", progress=100, message="处理完成，可以预览和下载。")
    except Exception as exc:
        traceback.print_exc()
        job.update(status="failed", message=f"处理失败：{exc}")


def process_source_separation_job(
    job_id: str, input_path: Path, video_output: Path, audio_output: Path
) -> None:
    job = jobs[job_id]
    try:
        with processing_lock:
            job.update(status="processing", progress=10, message="正在导出无声视频…")
            separate_audio_video(input_path, video_output, audio_output)
        job.update(status="completed", progress=100, message="音频和无声视频已分离，可以预览和下载。")
    except Exception as exc:
        traceback.print_exc()
        job.update(status="failed", message=f"音视频分离失败：{exc}")


def process_source_portrait_job(
    job_id: str, input_path: Path, output_path: Path, portrait_mode: str
) -> None:
    job = jobs[job_id]
    try:
        with processing_lock:
            job.update(status="processing", message="正在加载人像分离模型…")
            analyzer.render_portrait(input_path, output_path, portrait_mode, job)
        job.update(status="completed", progress=100, message="人像与背景分离完成，可以预览和下载。")
    except Exception as exc:
        traceback.print_exc()
        job.update(status="failed", message=f"人像与背景分离失败：{exc}")


@app.post("/api/jobs")
async def create_job(background_tasks: BackgroundTasks, mode: str = Form(...), video: UploadFile = File(...)) -> dict[str, str]:
    if mode not in MODES:
        raise HTTPException(400, "未知的处理模式")
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(400, "请上传 MP4、MOV、AVI、MKV、WebM 或 M4V 视频")
    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{suffix}"
    size = 0
    try:
        with input_path.open("wb") as destination:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "视频超过 500 MB 限制")
                destination.write(chunk)
    except Exception:
        input_path.unlink(missing_ok=True)
        raise
    finally:
        await video.close()
    output_path = RESULT_DIR / f"{job_id}.mp4"
    jobs[job_id] = {"status": "queued", "progress": 0, "message": "已排队", "mode": mode, "output": str(output_path)}
    background_tasks.add_task(process_job, job_id, input_path, output_path, mode)
    return {"job_id": job_id}


def validate_video_url(source_url: str) -> str:
    source_url = source_url.strip()
    # Mobile apps often copy a whole share message. Prefer the first embedded
    # URL so users can paste the message without manually extracting it.
    candidates = re.findall(r"https?://[^\s<>\"'，。！？、）】》]+", source_url)
    if candidates:
        source_url = candidates[0].rstrip(".,;:!?)]}）】》")
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        if "抖音" in source_url or re.search(r"\b[A-Za-z0-9@._-]{3,}:/", source_url):
            raise ValueError(
                "识别到抖音 App 口令，但其中不含可下载链接。请在抖音点“分享 → 复制链接”，"
                "然后粘贴包含 https://v.douyin.com/ 的完整分享文案。"
            )
        raise ValueError("请输入完整的 http 或 https 视频链接。")
    return source_url


async def update_cookie_store(cookie_file: UploadFile | None) -> bool:
    """Persist a Netscape cookies.txt locally for future URL downloads."""
    if cookie_file is None or not cookie_file.filename:
        return False
    try:
        if Path(cookie_file.filename).suffix.lower() not in {"", ".txt"}:
            raise HTTPException(400, "Cookie 文件必须是 .txt 格式")
        payload = await cookie_file.read(MAX_COOKIE_BYTES + 1)
        if not payload:
            raise HTTPException(400, "Cookie 文件为空")
        if len(payload) > MAX_COOKIE_BYTES:
            raise HTTPException(413, "Cookie 文件不能超过 5 MB")
        if b"\t" not in payload:
            raise HTTPException(400, "请上传 Netscape 格式的 cookies.txt")
        temporary = COOKIE_DIR / f"{uuid.uuid4().hex}.tmp"
        temporary.write_bytes(payload)
        temporary.replace(COOKIE_STORE)
        return True
    finally:
        await cookie_file.close()


def copy_saved_cookie_for_job(job_id: str) -> Path | None:
    """Use a disposable copy so the saved Cookie is never passed to cleanup."""
    if not COOKIE_STORE.is_file() or COOKIE_STORE.stat().st_size == 0:
        return None
    temporary = COOKIE_DIR / f"{job_id}.txt"
    shutil.copyfile(COOKIE_STORE, temporary)
    return temporary


def download_video_from_url(
    source_url: str, job_id: str, job: dict[str, Any], cookie_path: Path | None = None
) -> tuple[Path, Path]:
    """Download a single public video in a private job directory with yt-dlp."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("缺少 yt-dlp 依赖，请重新安装项目依赖。") from exc

    download_dir = URL_DOWNLOAD_DIR / job_id
    download_dir.mkdir(parents=True, exist_ok=False)
    job.update(status="downloading", progress=0, message="正在从视频链接下载…")
    def on_progress(event: dict[str, Any]) -> None:
        if event.get("status") != "downloading":
            return
        total = event.get("total_bytes") or event.get("total_bytes_estimate")
        downloaded = event.get("downloaded_bytes", 0)
        if total:
            job.update(progress=min(95, round(downloaded / total * 95, 1)), message="正在下载视频…")

    options = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(download_dir / "source.%(ext)s"),
        "noplaylist": True,
        "max_filesize": MAX_UPLOAD_BYTES,
        "retries": 3,
        "fragment_retries": 3,
        "continuedl": True,
        "overwrites": False,
        "restrictfilenames": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "progress_hooks": [on_progress],
    }
    if cookie_path is not None:
        options["cookiefile"] = str(cookie_path)
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.extract_info(source_url, download=True)

    candidates = [
        path for path in download_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and path.stat().st_size > 0
    ]
    if not candidates:
        raise RuntimeError("链接下载完成，但没有得到可处理的视频文件。")
    input_path = max(candidates, key=lambda path: path.stat().st_mtime)
    if input_path.stat().st_size > MAX_UPLOAD_BYTES:
        raise RuntimeError("下载的视频超过 500 MB 限制。")
    return input_path, download_dir


@app.post("/api/downloads")
async def create_download_preview(
    background_tasks: BackgroundTasks,
    source_url: str = Form(...),
    cookie_file: UploadFile | None = File(None),
) -> dict[str, str]:
    """Download a URL first, then let the user preview it before processing."""
    try:
        source_url = validate_video_url(source_url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    job_id = uuid.uuid4().hex
    cookie_updated = await update_cookie_store(cookie_file)
    cookie_path = copy_saved_cookie_for_job(job_id)
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "下载任务已排队",
        "source_url": source_url,
        "kind": "download_preview",
        "cookie_path": str(cookie_path) if cookie_path else None,
        "using_cookies": bool(cookie_path),
        "cookie_updated": cookie_updated,
    }
    background_tasks.add_task(download_preview_job, job_id, source_url, cookie_path)
    return {"job_id": job_id}


def download_preview_job(job_id: str, source_url: str, cookie_path: Path | None = None) -> None:
    job = jobs[job_id]
    try:
        input_path, download_dir = download_video_from_url(source_url, job_id, job, cookie_path)
        register_source(job_id, input_path, input_path.name, "download", download_dir)
        job.update(
            status="ready",
            progress=100,
            message="视频已下载，可以预览并选择处理工具。",
            source_path=str(input_path),
            source_dir=str(download_dir),
        )
    except Exception as exc:
        traceback.print_exc()
        job.update(status="failed", message=f"下载失败：{exc}")
    finally:
        if cookie_path is not None:
            cookie_path.unlink(missing_ok=True)


@app.post("/api/downloads/{job_id}/process")
async def process_download_preview(
    job_id: str, background_tasks: BackgroundTasks, mode: str = Form(...)
) -> dict[str, str]:
    if mode not in MODES:
        raise HTTPException(400, "未知的处理模式")
    job = jobs.get(job_id)
    if not job or job.get("kind") != "download_preview":
        raise HTTPException(404, "下载任务不存在")
    if job.get("status") != "ready":
        raise HTTPException(409, "请等待视频下载完成后再开始处理")
    input_path = Path(str(job.get("source_path", "")))
    if not input_path.is_file():
        raise HTTPException(404, "下载的视频文件已不存在")
    output_path = RESULT_DIR / f"{job_id}.mp4"
    job.update(status="queued", progress=0, message="处理任务已排队", mode=mode, output=str(output_path))
    background_tasks.add_task(process_downloaded_preview, job_id, input_path, output_path, mode)
    return {"job_id": job_id}


def process_downloaded_preview(job_id: str, input_path: Path, output_path: Path, mode: str) -> None:
    job = jobs[job_id]
    try:
        with processing_lock:
            job.update(status="processing", message="正在加载分析模型…")
            analyzer.render(input_path, output_path, mode, job)
        job.update(status="completed", progress=100, message="处理完成，可以预览和下载。")
    except Exception as exc:
        traceback.print_exc()
        job.update(status="failed", message=f"处理失败：{exc}")
    finally:
        source_dir = Path(str(job.get("source_dir", "")))
        if source_dir.is_dir():
            shutil.rmtree(source_dir, ignore_errors=True)


@app.get("/api/downloads/{job_id}/preview")
def preview_download(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    path = Path(str(job.get("source_path", ""))) if job else None
    if not job or job.get("status") != "ready" or not path.is_file():
        raise HTTPException(404, "可预览的视频不存在")
    return FileResponse(path, media_type="video/mp4", filename=f"download-{job_id[:8]}{path.suffix}")


@app.post("/api/jobs/from-url")
async def create_job_from_url(
    background_tasks: BackgroundTasks,
    mode: str = Form(...),
    source_url: str = Form(...),
) -> dict[str, str]:
    if mode not in MODES:
        raise HTTPException(400, "未知的处理模式")
    try:
        source_url = validate_video_url(source_url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    job_id = uuid.uuid4().hex
    output_path = RESULT_DIR / f"{job_id}.mp4"
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "下载任务已排队",
        "mode": mode,
        "source_url": source_url,
        "output": str(output_path),
    }
    background_tasks.add_task(process_url_job, job_id, source_url, output_path, mode)
    return {"job_id": job_id}


def process_job(job_id: str, input_path: Path, output_path: Path, mode: str) -> None:
    job = jobs[job_id]
    try:
        with processing_lock:
            job.update(status="processing", message="正在加载分析模型…")
            analyzer.render(input_path, output_path, mode, job)
        job.update(status="completed", progress=100, message="处理完成，可以预览和下载。")
    except Exception as exc:
        traceback.print_exc()
        job.update(status="failed", message=f"处理失败：{exc}")
    finally:
        input_path.unlink(missing_ok=True)


def process_url_job(job_id: str, source_url: str, output_path: Path, mode: str) -> None:
    job = jobs[job_id]
    download_dir: Path | None = None
    try:
        input_path, download_dir = download_video_from_url(source_url, job_id, job)
        with processing_lock:
            job.update(status="processing", message="下载完成，正在加载分析模型…")
            analyzer.render(input_path, output_path, mode, job)
        job.update(status="completed", progress=100, message="处理完成，可以预览和下载。")
    except Exception as exc:
        traceback.print_exc()
        job.update(status="failed", message=f"处理失败：{exc}")
    finally:
        if download_dir is not None:
            shutil.rmtree(download_dir, ignore_errors=True)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在或服务已重启")
    response = {
        key: value for key, value in job.items()
        if key not in {"output", "outputs", "source_path", "source_dir", "cookie_path"}
    }
    if job.get("kind") == "download_preview" and job["status"] == "ready":
        response["preview_url"] = f"/api/downloads/{job_id}/preview"
    if job["status"] == "completed":
        if job.get("tool") == "separate":
            response["downloads"] = {
                "video": f"/api/jobs/{job_id}/download/video",
                "audio": f"/api/jobs/{job_id}/download/audio",
            }
        else:
            response["download_url"] = f"/api/jobs/{job_id}/download"
    return response


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(404, "处理结果不存在")
    path = Path(job["output"])
    if not path.exists():
        raise HTTPException(404, "处理结果文件丢失")
    if path.suffix.lower() == ".webm":
        return FileResponse(path, media_type="video/webm", filename=f"portrait-transparent-{job_id[:8]}.webm")
    return FileResponse(path, media_type="video/mp4", filename=f"deep-analysis-{job_id[:8]}.mp4")


@app.get("/api/jobs/{job_id}/download/{artifact}")
def download_job_artifact(job_id: str, artifact: str) -> FileResponse:
    job = jobs.get(job_id)
    outputs = job.get("outputs") if job else None
    if not job or job.get("status") != "completed" or not isinstance(outputs, dict):
        raise HTTPException(404, "处理结果不存在")
    if artifact not in {"video", "audio"} or artifact not in outputs:
        raise HTTPException(404, "请求的导出文件不存在")
    path = Path(str(outputs[artifact]))
    if not path.is_file():
        raise HTTPException(404, "导出文件已丢失")
    if artifact == "video":
        return FileResponse(path, media_type="video/mp4", filename=f"silent-video-{job_id[:8]}.mp4")
    return FileResponse(path, media_type="audio/mpeg", filename=f"audio-track-{job_id[:8]}.mp3")
