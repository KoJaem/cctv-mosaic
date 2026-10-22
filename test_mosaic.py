import builtins
import os
import tempfile

import cv2
import numpy as np

import mosaic
from mosaic import pixelate, process

# choose_video: 영상만 목록에, 잘못된 입력은 재질문
d = tempfile.mkdtemp()
for f in ["b.mp4", "a.MOV", "note.txt"]:
    open(os.path.join(d, f), "w").close()
mosaic.TARGET = d
answers = iter(["x", "9", "2"])
builtins.input = lambda _: next(answers)
assert mosaic.choose_video() == os.path.join(d, "b.mp4")
answers = iter(["0", "2"])
assert mosaic.choose_mode() == "keep"

# confirm_runs: min_frames 미만 연속 구간은 버림 (= 가림)
assert mosaic.confirm_runs([0, 0, None, 1, 1, 1, None, 2], 3) == [None, None, None, 1, 1, 1, None, None]
assert mosaic.confirm_runs([None, None], 1) == [None, None]

# intervals / iou
assert mosaic.intervals([False, True, True, False, True], 1) == ["00:01~00:02", "00:04~00:04"]
assert mosaic.iou((0, 0, 10, 10), (0, 0, 10, 10)) > 0.99
assert mosaic.iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0

# load_keep_feats: 얼굴 없는 사진·사진 아닌 파일은 건너뜀
mosaic.KEEP = tempfile.mkdtemp()
cv2.imwrite(os.path.join(mosaic.KEEP, "noface.jpg"), np.zeros((100, 100, 3), np.uint8))
open(os.path.join(mosaic.KEEP, "memo.txt"), "w").close()
assert mosaic.load_keep_feats(0.6) == []

# pixelate: 박스 안은 blocks*blocks 이하 색, 박스 밖은 그대로
img = np.random.randint(0, 255, (100, 100, 3), np.uint8)
orig = img.copy()
pixelate(img, 10, 10, 40, 40, blocks=4)
assert len(np.unique(img[10:50, 10:50].reshape(-1, 3), axis=0)) <= 16
assert (img[60:, 60:] == orig[60:, 60:]).all()
pixelate(img, -20, -20, 10, 10)  # 화면 밖 박스는 무시

# 전체 파이프라인: 합성 영상 → 출력 생성, 프레임 수 유지
d = tempfile.mkdtemp()
src, dst = os.path.join(d, "in.mp4"), os.path.join(d, "out.mp4")
w = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
for _ in range(20):
    w.write(np.random.randint(0, 255, (120, 160, 3), np.uint8))
w.release()
process(src, dst)
assert int(cv2.VideoCapture(dst).get(cv2.CAP_PROP_FRAME_COUNT)) == 20

# best_match: 가짜 얼굴 행으로 인식 API 호출 경로 확인 (크기 미달 얼굴은 비교 없이 가림)
rec = mosaic.recognizer()
frame = np.random.randint(0, 255, (200, 200, 3), np.uint8)
row = np.array([50, 50, 80, 80, 70, 80, 110, 80, 90, 100, 75, 115, 105, 115, 0.9], np.float32)
feat = rec.feature(rec.alignCrop(frame, row))
assert mosaic.best_match(rec, frame, [row], [feat], 0.45, 40) == 0
assert mosaic.best_match(rec, frame, [row], [feat], 0.45, 100) is None
assert mosaic.best_match(rec, frame, [row], [-feat], 0.45, 40) is None

# 특정 인물 제외 모드 + 확대 감지 경로도 끝까지 돌아가는지
process(src, dst, scale=2.0, keep_feats=[np.zeros((1, 128), np.float32)])
assert int(cv2.VideoCapture(dst).get(cv2.CAP_PROP_FRAME_COUNT)) == 20

# 얼굴 추출: 얼굴 없는 영상이면 저장 0장
mosaic.FACES = tempfile.mkdtemp()
mosaic.extract_faces(src)
assert os.listdir(os.path.join(mosaic.FACES, "in")) == []
print("OK")
