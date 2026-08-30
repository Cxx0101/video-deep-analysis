from __future__ import annotations

import json
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import shutil
import subprocess
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from vendor_midas.midas_net_custom import MidasNet_small


# The web service keeps all mutable files inside the project directory.
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR
DATA_DIR = STORAGE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
URL_DOWNLOAD_DIR = DATA_DIR / "url_downloads"
MODEL_DIR = STORAGE_DIR / "models"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
for directory in (UPLOAD_DIR, RESULT_DIR, URL_DOWNLOAD_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Keep all downloaded model files beside the project.

MODES = {
    "depth": "灰度深度图",
    "pose": "人体姿态骨架叠加",
    "depth_pose": "深度图 + 人体姿态骨架",
    "face": "面部 478 点网格",
    "all": "深度图 + 人体姿态 + 面部网格",
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}

jobs: dict[str, dict[str, Any]] = {}
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


class VideoAnalyzer:
    """Lazily loads the depth and landmark models exactly once per process."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model: Any | None = None
        self.depth_ort: Any | None = None
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.drawer = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles

    @property
    def device_name(self) -> str:
        if self.depth_ort is not None and "CUDAExecutionProvider" in self.depth_ort.get_providers():
            return "NVIDIA CUDA (ONNX Runtime)"
        if self.device.type == "cuda":
            return f"NVIDIA GPU: {torch.cuda.get_device_name(0)}"
        return "CPU"

    def load_depth_model(self) -> None:
        if self.depth_model is not None:
            return
        # MiDaS Small is an official monocular-depth model. The single required
        # weight is downloaded into the project directory on first use.
        with models_lock:
            if self.depth_model is not None:
                return
            weights = MODEL_DIR / "midas_v21_small_256.pt"
            if not weights.exists():
                from urllib.request import urlretrieve

                urlretrieve("https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt", weights)
            self.depth_model = MidasNet_small(path=str(weights))
            self.depth_model.to(self.device).eval()
            if "CUDAExecutionProvider" in ort.get_available_providers():
                onnx_path = MODEL_DIR / "midas_v21_small_256.onnx"
                if not onnx_path.exists():
                    sample = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
                    torch.onnx.export(self.depth_model.cpu(), sample, str(onnx_path), input_names=["image"], output_names=["depth"], opset_version=17, dynamo=False)
                self.depth_ort = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def depth_frame(self, rgb: np.ndarray) -> np.ndarray:
        self.load_depth_model()
        assert self.depth_model is not None
        resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        batch_array = np.ascontiguousarray(((resized - 0.5) / 0.5).transpose(2, 0, 1)[None])
        if self.depth_ort is not None:
            prediction = self.depth_ort.run(None, {"image": batch_array})[0][0]
            prediction = cv2.resize(prediction, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
            values = prediction
        else:
            with torch.inference_mode():
                batch = torch.from_numpy(batch_array).to(self.device)
                prediction = self.depth_model(batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False
                ).squeeze()
                values = prediction.detach().float().cpu().numpy()
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


analyzer = VideoAnalyzer()
app = FastAPI(title="视频深度分析工具")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": analyzer.device_name}


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
    parsed = urlparse(source_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("请输入完整的 http 或 https 视频链接。")
    return source_url.strip()


def download_video_from_url(source_url: str, job_id: str, job: dict[str, Any]) -> tuple[Path, Path]:
    """Download a single public video in a private job directory with yt-dlp."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("缺少 yt-dlp 依赖，请重新安装项目依赖。") from exc

    download_dir = URL_DOWNLOAD_DIR / job_id
    download_dir.mkdir(parents=True, exist_ok=False)
    job.update(status="downloading", progress=0, message="正在从视频链接下载…")
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
    }
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
    response = {key: value for key, value in job.items() if key != "output"}
    if job["status"] == "completed":
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
    return FileResponse(path, media_type="video/mp4", filename=f"deep-analysis-{job_id[:8]}.mp4")
