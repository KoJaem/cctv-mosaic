# CCTV 얼굴 모자이크

영상에서 얼굴을 자동으로 찾아 모자이크 처리합니다. 모든 처리는 이 컴퓨터 안에서만 이루어지며, 영상이 외부로 전송되지 않습니다. 인터넷 연결 없이도 동작합니다.

## 실행

1. 처리할 영상을 `target/` 폴더에 넣습니다.
2. 실행합니다.

```bash
cd ~/Documents/cctv-mosaic
python3 mosaic.py
```

3. 목록에서 번호를 입력합니다.

```
/Users/kojaemin/Documents/cctv-mosaic/target 영상 목록:
  1. 정문_1022.mp4
  2. 주차장_1022.mp4
모자이크할 영상 번호: 2
```

결과는 `result/주차장_1022_mosaic.mp4`로 저장됩니다. 같은 이름의 결과가 있으면 덮어씁니다.

`target/` 밖의 파일을 처리하거나 출력 위치를 지정하려면 경로를 직접 넣습니다.

```bash
python3 mosaic.py ~/"Desktop/CCTV 영상.mp4"            # 결과는 result/ 에
python3 mosaic.py ~/"Desktop/CCTV 영상.mp4" 결과.mp4   # 결과 위치 지정
```

경로에 공백이 있으면 따옴표로 감싸고, `~`는 따옴표 밖에 둡니다. 터미널 창에 파일을 끌어다 놓으면 경로가 자동으로 입력됩니다.

## 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--threshold` | 0.6 | 감지 기준. 낮출수록 더 많이 가립니다 (얼굴이 아닌 곳도 가려질 수 있음) |
| `--scale` | 1.0 | 감지용 확대 배율. 얼굴이 작게 찍혔으면 `2.0` |
| `--pad` | 0.4 | 얼굴 박스를 넓히는 비율 (머리·귀까지 가림) |
| `--hold` | 15 | 감지된 위치를 앞뒤 몇 프레임까지 계속 가릴지 |
| `--blocks` | 8 | 모자이크 칸 수. 작을수록 거칠게 |

얼굴이 빠진 장면이 있을 때:

```bash
python3 mosaic.py --threshold 0.4 --scale 2.0
```

## 결과물 특징

- 오디오는 제거됩니다 (목소리도 개인정보).
- 메타데이터(촬영 정보 등)는 제거됩니다.
- 영상에 직접 박혀 있는 날짜·시간 표시는 그대로 남습니다.

## ⚠️ 전달 전 반드시 확인

자동 감지는 완벽하지 않습니다. **결과 영상 전체를 직접 재생해서 확인한 뒤 전달하세요.**

- 뒷모습·옆모습, 저화질, 어두운 장면, 멀리 작게 찍힌 사람은 놓칠 수 있습니다.
- 번호판, 이름표, 화면 속 문서 등 얼굴이 아닌 식별 정보는 가리지 않습니다.

## 설치 (가상환경 `.venv`)

이 Mac에는 필요한 패키지가 이미 설치돼 있어 위 실행 방법대로 바로 쓸 수 있습니다. 다른 컴퓨터에서 쓰거나, 시스템 Python을 건드리지 않고 이 폴더 전용 환경을 만들고 싶을 때 아래처럼 가상환경을 사용합니다.

### 처음 한 번 (인터넷 필요)

```bash
cd ~/Documents/cctv-mosaic
python3 -m venv .venv              # 가상환경 생성
source .venv/bin/activate          # 가상환경 켜기
pip install opencv-python          # numpy도 함께 설치됨
brew install ffmpeg                # 선택. 없어도 동작하지만 결과 파일이 일부 플레이어에서 재생 안 될 수 있음
```

### 사용할 때마다

```bash
cd ~/Documents/cctv-mosaic
source .venv/bin/activate          # 프롬프트 앞에 (.venv) 표시되면 켜진 것
python mosaic.py
deactivate                         # 다 쓰면 끄기
```

켜고 끄기 없이 한 줄로 실행할 수도 있습니다.

```bash
.venv/bin/python mosaic.py
```

Windows에서는 `source .venv/bin/activate` 대신 `.venv\Scripts\activate`를 사용합니다.

### 참고

- 실행 중 `[ WARN:0@...] ... setPreferableTarget ...` 줄이 보일 수 있습니다. OpenCV 5.x의 안내 메시지이며 결과에는 영향 없습니다.
- `.venv` 폴더는 다른 컴퓨터로 복사해도 동작하지 않습니다. 새 컴퓨터에서는 `.venv`를 빼고 복사한 뒤 위 "처음 한 번" 과정을 다시 하세요.
- `.venv`가 꼬였으면 `rm -rf .venv` 후 다시 만들면 됩니다.

## 다른 컴퓨터로 옮기기

`.venv`, `result/`, `target/` 안의 영상을 제외하고 폴더를 복사합니다. `mosaic.py`와 `face_detection_yunet_2023mar.onnx`는 반드시 같은 폴더에 있어야 합니다. 모델 파일이 없으면 인터넷에서 다시 받으려고 시도합니다.

## 동작 확인 테스트

```bash
python3 test_mosaic.py   # "OK" 출력되면 정상
```
