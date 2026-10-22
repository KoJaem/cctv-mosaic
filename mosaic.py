#!/usr/bin/env python3
"""CCTV 영상 얼굴 모자이크. 영상은 이 컴퓨터 안에서만 처리됨 (외부 전송 없음).

사용: python3 mosaic.py              # 작업(전체/인물 제외/얼굴 추출)과 target/ 영상 선택, 결과는 result/
      python3 mosaic.py 입력.mp4 [출력.mp4] [--mode all|keep|extract] [--threshold 0.6] [--scale 1.0] ...
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

import cv2

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
DET_MODEL = ("face_detection_yunet", "face_detection_yunet_2023mar.onnx")
REC_MODEL = ("face_recognition_sface", "face_recognition_sface_2021dec.onnx")
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "target")
RESULT = os.path.join(HERE, "result")
FACES = os.path.join(HERE, "faces")
KEEP = os.path.join(HERE, "keep")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".ts", ".m4v", ".h264", ".dav")
IMG_EXTS = (".jpg", ".jpeg", ".png")
MODES = [("all", "전체 얼굴 모자이크"),
         ("keep", "특정 인물 제외하고 모자이크 (keep/ 사진 사용)"),
         ("extract", "얼굴 추출 (keep/ 에 넣을 사진 고르기용)")]


def ask(prompt, n):
    """1~n 번호를 입력받아 0부터 시작하는 인덱스로 반환."""
    while True:
        try:
            s = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\n취소됨.")
        if s.isdigit() and 1 <= int(s) <= n:
            return int(s) - 1
        print(f"1~{n} 사이 번호를 입력하세요.")


def choose_mode():
    print("작업 선택:")
    for i, (_, desc) in enumerate(MODES, 1):
        print(f"  {i}. {desc}")
    return MODES[ask("작업 번호: ", len(MODES))][0]


def choose_video():
    """target/ 영상 목록을 보여주고 번호로 선택받음."""
    os.makedirs(TARGET, exist_ok=True)
    files = sorted(f for f in os.listdir(TARGET) if f.lower().endswith(VIDEO_EXTS))
    if not files:
        sys.exit(f"영상 없음. {TARGET} 에 영상 파일을 넣고 다시 실행하세요.")
    print(f"{TARGET} 영상 목록:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")
    return os.path.join(TARGET, files[ask("영상 번호: ", len(files))])


def model_path(model):
    # 모델 파일만 다운로드함. 영상·사진은 업로드하지 않음.
    subdir, name = model
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        print(f"모델 다운로드 중 (최초 1회): {name}")
        urllib.request.urlretrieve(ZOO + subdir + "/" + name, path + ".part")
        os.replace(path + ".part", path)
    return path


def detector(size, threshold):
    return cv2.FaceDetectorYN.create(model_path(DET_MODEL), "", size, threshold)


def recognizer():
    return cv2.FaceRecognizerSF.create(model_path(REC_MODEL), "")


def pixelate(img, x, y, w, h, blocks=8):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    small = cv2.resize(img[y0:y1, x0:x1], (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    img[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def detect_rows(det, frame, scale):
    """YuNet 결과 행(박스 4 + 랜드마크 10 + 점수 1)을 원본 좌표로."""
    if scale != 1.0:
        frame = cv2.resize(frame, (det.getInputSize()[0], det.getInputSize()[1]))
    _, faces = det.detect(frame)
    rows = [] if faces is None else [f.copy() for f in faces]
    for f in rows:
        f[:14] /= scale
    return rows


def load_keep_feats(threshold):
    """keep/ 사진마다 가장 큰 얼굴의 특징 벡터."""
    os.makedirs(KEEP, exist_ok=True)
    rec, feats = None, []
    for name in sorted(os.listdir(KEEP)):
        if not name.lower().endswith(IMG_EXTS):
            continue
        img = cv2.imread(os.path.join(KEEP, name))
        if img is None:
            print(f"  경고: {name} 읽기 실패, 건너뜀")
            continue
        if max(img.shape[:2]) < 200:  # 작은 얼굴 사진은 확대해야 감지가 잘 됨
            img = cv2.resize(img, None, fx=2, fy=2)
        _, faces = detector((img.shape[1], img.shape[0]), threshold).detect(img)
        if faces is None:
            print(f"  경고: {name} 에서 얼굴을 못 찾음, 건너뜀")
            continue
        rec = rec or recognizer()
        f = max(faces, key=lambda r: r[2] * r[3])
        feats.append(rec.feature(rec.alignCrop(img, f)))
    return feats


def best_match(rec, frame, rows, keep_feats, match, min_face):
    """keep 인물로 볼 얼굴 인덱스 (프레임당 최대 1개). 애매하면 None = 가림."""
    best, best_sim = None, match
    for i, f in enumerate(rows):
        if min(f[2], f[3]) < min_face:
            continue
        feat = rec.feature(rec.alignCrop(frame, f))
        sim = max(rec.match(feat, k, cv2.FaceRecognizerSF_FR_COSINE) for k in keep_feats)
        if sim >= best_sim:
            best, best_sim = i, sim
    return best


def confirm_runs(cand, min_frames):
    """후보가 min_frames 프레임 이상 연속된 구간만 남기고 나머지는 None (= 가림)."""
    out = [None] * len(cand)
    i = 0
    while i < len(cand):
        j = i
        while j < len(cand) and cand[j] is not None:
            j += 1
        if j - i >= min_frames:
            out[i:j] = cand[i:j]
        i = j + 1
    return out


def ts(sec):
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def intervals(flags, fps):
    """True인 프레임 구간을 'MM:SS~MM:SS' 목록으로."""
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append(f"{ts(start / fps)}~{ts((i - 1) / fps)}")
            start = None
    return out


def open_video(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"영상 열기 실패: {src}")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, cap.get(cv2.CAP_PROP_FPS) or 30, W, H


def extract_faces(src, threshold=0.6, scale=1.0, min_face=40, pad=0.5):
    """1초에 1프레임씩 얼굴을 잘라 faces/<영상이름>/ 에 저장 (keep 사진 고르기용)."""
    out_dir = os.path.join(FACES, os.path.splitext(os.path.basename(src))[0])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(KEEP, exist_ok=True)
    cap, fps, W, H = open_video(src)
    det = detector((int(W * scale), int(H * scale)), threshold)
    step = max(1, round(fps))
    i = saved = 0
    while cap.grab():
        if i % step == 0:
            _, frame = cap.retrieve()
            for k, f in enumerate(detect_rows(det, frame, scale)):
                x, y, w, h = f[:4]
                if min(w, h) < min_face:
                    continue
                px, py = w * pad, h * pad
                crop = frame[max(int(y - py), 0):int(y + h + py), max(int(x - px), 0):int(x + w + px)]
                t = i / fps
                cv2.imwrite(os.path.join(out_dir, f"{int(t // 60):02d}m{int(t % 60):02d}s_{k}.jpg"), crop)
                saved += 1
        i += 1
    cap.release()
    print(f"얼굴 사진 {saved}장 저장: {out_dir}")
    if saved:
        print(f"이 중 모자이크하지 않을 인물 사진 3~10장을 {KEEP} 에 복사한 뒤, 2번(특정 인물 제외)을 실행하세요.")
    else:
        print(f"저장된 얼굴 없음. --scale 2.0 또는 --threshold 0.4 로 다시 시도하세요. (얼굴이 {min_face}px 미만이면 저장 안 함)")


def detect_all(src, threshold, scale, keep_feats=None, match=0.45, min_face=40):
    """1차 패스: 프레임별 얼굴 박스 목록, 프레임별 keep 후보 인덱스."""
    cap, _, W, H = open_video(src)
    det = detector((int(W * scale), int(H * scale)), threshold)
    rec = recognizer() if keep_feats else None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_boxes, cand = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rows = detect_rows(det, frame, scale)
        all_boxes.append([f[:4] for f in rows])
        cand.append(best_match(rec, frame, rows, keep_feats, match, min_face) if rec else None)
        if len(all_boxes) % 100 == 0:
            print(f"\r감지 {len(all_boxes)}/{total}", end="", flush=True)
    cap.release()
    print()
    return all_boxes, cand


def process(src, dst, threshold=0.6, pad=0.4, hold=15, scale=1.0, blocks=8,
            keep_feats=None, match=0.45, min_face=40, min_frames=5):
    all_boxes, cand = detect_all(src, threshold, scale, keep_feats, match, min_face)
    keep = confirm_runs(cand, min_frames)
    n = len(all_boxes)

    # 2차 패스: 앞뒤 hold 프레임에서 감지된 박스까지 모두 가림 (감지가 깜빡이며 빠지는 프레임 보완).
    # 단, 이 프레임에서 확정된 keep 얼굴과 그 자리에 겹치는 이웃 프레임 박스는 가리지 않음.
    cap, fps, W, H = open_video(src)
    tmp = dst + ".tmp.mp4"
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        kbox = None if keep[i] is None else all_boxes[i][keep[i]]
        for j in range(max(0, i - hold), min(n, i + hold + 1)):
            for b, (x, y, w, h) in enumerate(all_boxes[j]):
                if kbox is not None and ((j == i and b == keep[i]) or (j != i and iou((x, y, w, h), kbox) > 0.3)):
                    continue
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
    if keep_feats:
        spans = intervals([k is not None for k in keep], fps)
        if spans:
            print(f"모자이크 제외 구간: {', '.join(spans)}")
            print("이 구간에서 해당 인물만 노출되고 다른 사람은 가려졌는지 꼭 확인하세요.")
        else:
            print("해당 인물이 확실히 인식된 구간 없음 — 전원 모자이크됨.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="영상 얼굴 모자이크 (로컬 처리)")
    p.add_argument("src", nargs="?", help="생략하면 target/ 에서 선택")
    p.add_argument("dst", nargs="?", help="생략하면 result/<이름>_mosaic.mp4")
    p.add_argument("--mode", choices=[m for m, _ in MODES], help="all: 전체 / keep: 특정 인물 제외 / extract: 얼굴 추출. 생략하면 물어봄")
    p.add_argument("--threshold", type=float, default=0.6, help="감지 기준. 낮출수록 더 많이 가림 (오탐 증가)")
    p.add_argument("--pad", type=float, default=0.4, help="박스 확장 비율 (머리·귀 포함)")
    p.add_argument("--hold", type=int, default=15, help="감지 전후 몇 프레임까지 유지할지")
    p.add_argument("--scale", type=float, default=1.0, help="감지용 확대 배율. 얼굴 작으면 2.0")
    p.add_argument("--blocks", type=int, default=8, help="모자이크 칸 수. 작을수록 거칠게")
    p.add_argument("--match", type=float, default=0.45, help="keep 인물로 판단할 유사도. 높일수록 엄격")
    p.add_argument("--min-face", type=int, default=40, help="이보다 작은 얼굴(px)은 인식하지 않고 무조건 가림")
    p.add_argument("--min-frames", type=int, default=5, help="연속 몇 프레임 이상 인식돼야 모자이크 제외할지")
    a = p.parse_args()

    mode = a.mode or choose_mode()
    keep_feats = None
    if mode == "keep":
        keep_feats = load_keep_feats(a.threshold)
        if not keep_feats:
            sys.exit(f"{KEEP} 에 쓸 수 있는 인물 사진이 없음. 먼저 3번(얼굴 추출)으로 사진을 뽑아 넣으세요.")
        print(f"keep/ 사진 {len(keep_feats)}장 — 이 인물은 모자이크 제외")

    src = a.src or choose_video()
    if mode == "extract":
        extract_faces(src, a.threshold, a.scale, a.min_face)
        sys.exit()

    dst = a.dst
    if not dst:
        os.makedirs(RESULT, exist_ok=True)
        suffix = "_mosaic_keep.mp4" if keep_feats else "_mosaic.mp4"
        dst = os.path.join(RESULT, os.path.splitext(os.path.basename(src))[0] + suffix)
    process(src, dst, a.threshold, a.pad, a.hold, a.scale, a.blocks,
            keep_feats, a.match, a.min_face, a.min_frames)
