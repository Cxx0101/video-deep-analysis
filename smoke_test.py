"""Local HTTP end-to-end verification for all five user-facing modes."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import requests


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "smoke-input.mp4"
CHECKS = ROOT / "data" / "smoke-results"
BASE_URL = "http://127.0.0.1:8000"
MODES = ["depth", "pose", "depth_pose", "face", "all"]


def make_short_video() -> None:
    INPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(INPUT), cv2.VideoWriter_fourcc(*"mp4v"), 8, (320, 240))
    for frame_index in range(3):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:] = (30 + frame_index * 18, 50, 90)
        # A moving face-like subject ensures the landmark pipelines receive valid RGB frames.
        center = (160 + frame_index * 8, 105)
        cv2.circle(image, center, 54, (185, 165, 140), -1)
        cv2.circle(image, (center[0] - 19, center[1] - 8), 7, (30, 30, 30), -1)
        cv2.circle(image, (center[0] + 19, center[1] - 8), 7, (30, 30, 30), -1)
        cv2.ellipse(image, (center[0], center[1] + 20), (20, 9), 0, 0, 180, (30, 30, 30), 3)
        cv2.rectangle(image, (center[0] - 42, 158), (center[0] + 42, 236), (110, 100, 80), -1)
        writer.write(image)
    writer.release()


def run_mode(mode: str) -> None:
    with INPUT.open("rb") as source:
        response = requests.post(BASE_URL + "/api/jobs", data={"mode": mode}, files={"video": (INPUT.name, source, "video/mp4")}, timeout=30)
    response.raise_for_status()
    job_id = response.json()["job_id"]
    for _ in range(240):
        state = requests.get(f"{BASE_URL}/api/jobs/{job_id}", timeout=10).json()
        if state["status"] == "completed":
            payload = requests.get(BASE_URL + state["download_url"], timeout=60)
            payload.raise_for_status()
            destination = CHECKS / f"{mode}.mp4"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload.content)
            capture = cv2.VideoCapture(str(destination))
            ok, frame = capture.read()
            capture.release()
            if not ok or frame is None or frame.shape[:2] != (240, 320):
                raise RuntimeError(f"{mode}: downloaded MP4 cannot be decoded")
            print(f"PASS {mode}: {destination.stat().st_size} bytes")
            return
        if state["status"] == "failed":
            raise RuntimeError(f"{mode}: {state['message']}")
        time.sleep(0.5)
    raise TimeoutError(f"{mode}: timed out")


if __name__ == "__main__":
    make_short_video()
    for selected_mode in MODES:
        run_mode(selected_mode)
    print("ALL FIVE MODES PASSED")
