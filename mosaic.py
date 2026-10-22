#!/usr/bin/env python3
"""CCTV 영상 얼굴 모자이크. 영상은 이 컴퓨터 안에서만 처리됨 (외부 전송 없음).

사용: python3 mosaic.py              # target/ 영상 중 선택, 결과는 result/
      python3 mosaic.py 입력.mp4 [출력.mp4] [--threshold 0.6] [--pad 0.4] [--hold 15] [--scale 1.0]
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

import cv2

MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "face_detection_yunet_2023mar.onnx")
TARGET = os.path.join(HERE, "target")
RESULT = os.path.join(HERE, "result")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".ts", ".m4v", ".h264", ".dav")


def choose_video():
    """target/ 영상 목록을 보여주고 번호로 선택받음."""
    os.makedirs(TARGET, exist_ok=True)
    files = sorted(f for f in os.listdir(TARGET) if f.lower().endswith(VIDEO_EXTS))
    if not files:
        sys.exit(f"영상 없음. {TARGET} 에 영상 파일을 넣고 다시 실행하세요.")
    print(f"{TARGET} 영상 목록:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")
    while True:
        try:
            s = input("모자이크할 영상 번호: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\n취소됨.")
        if s.isdigit() and 1 <= int(s) <= len(files):
            return os.path.join(TARGET, files[int(s) - 1])
        print(f"1~{len(files)} 사이 번호를 입력하세요.")


def model_path():
    # 모델 파일만 다운로드함. 영상은 업로드하지 않음.
    if not os.path.exists(MODEL_PATH):
        print("얼굴 감지 모델 다운로드 중 (최초 1회)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def pixelate(img, x, y, w, h, blocks=8):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    small = cv2.resize(img[y0:y1, x0:x1], (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    img[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)


def detect_all(src, threshold, scale):
    """1차 패스: 프레임별 얼굴 박스 목록."""
    cap = cv2.VideoCapture(src)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det = cv2.FaceDetectorYN.create(model_path(), "", (int(W * scale), int(H * scale)), threshold)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_boxes = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (int(W * scale), int(H * scale)))
        _, faces = det.detect(frame)
        all_boxes.append([] if faces is None else [f[:4] / scale for f in faces])
        if len(all_boxes) % 100 == 0:
            print(f"\r감지 {len(all_boxes)}/{total}", end="", flush=True)
    cap.release()
    print()
    return all_boxes


def process(src, dst, threshold=0.6, pad=0.4, hold=15, scale=1.0, blocks=8):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"영상 열기 실패: {src}")
    cap.release()

    all_boxes = detect_all(src, threshold, scale)
    n = len(all_boxes)

    # 2차 패스: 앞뒤 hold 프레임에서 감지된 박스까지 모두 가림 (감지가 깜빡이며 빠지는 프레임 보완)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = dst + ".tmp.mp4"
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        for bs in all_boxes[max(0, i - hold): i + hold + 1]:
            for x, y, w, h in bs:
                px, py = w * pad, h * pad
                pixelate(frame, int(x - px), int(y - py), int(w + 2 * px), int(h + 2 * py), blocks)
        out.write(frame)
    cap.release()
    out.release()

    # H.264 재인코딩 + 메타데이터 제거. 오디오(음성도 개인정보)는 넣지 않음.
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-an", "-map_metadata", "-1",
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", dst], check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, dst)

    hit = sum(1 for b in all_boxes if b)
    print(f"완료: {dst}")
    print(f"얼굴 감지된 프레임 {hit}/{n}. 자동 감지는 누락 가능 — 전달 전 전체 영상 직접 확인 필수.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="영상 얼굴 모자이크 (로컬 처리)")
    p.add_argument("src", nargs="?", help="생략하면 target/ 에서 선택")
    p.add_argument("dst", nargs="?", help="생략하면 result/<이름>_mosaic.mp4")
    p.add_argument("--threshold", type=float, default=0.6, help="감지 기준. 낮출수록 더 많이 가림 (오탐 증가)")
    p.add_argument("--pad", type=float, default=0.4, help="박스 확장 비율 (머리·귀 포함)")
    p.add_argument("--hold", type=int, default=15, help="감지 전후 몇 프레임까지 유지할지")
    p.add_argument("--scale", type=float, default=1.0, help="감지용 확대 배율. 얼굴 작으면 2.0")
    p.add_argument("--blocks", type=int, default=8, help="모자이크 칸 수. 작을수록 거칠게")
    a = p.parse_args()
    src = a.src or choose_video()
    dst = a.dst
    if not dst:
        os.makedirs(RESULT, exist_ok=True)
        dst = os.path.join(RESULT, os.path.splitext(os.path.basename(src))[0] + "_mosaic.mp4")
    process(src, dst, a.threshold, a.pad, a.hold, a.scale, a.blocks)
