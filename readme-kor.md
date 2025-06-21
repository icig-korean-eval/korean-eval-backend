# 외국인 대상 한국어 학습 프로그램 RestAPI 서버

- 사용자에게 모델을 제공하기 위한 RestAPI 서버
- 파인튜닝한 모델을 서빙
  - [외국인 한국어 발화 ASR 모델](https://huggingface.co/icig/non-native-korean-speech-asr)
  - [표준어 발화 IPA ASR 모델](https://huggingface.co/icig/normal-korean-ipa-translation)
  - [아나운서 발화 IPA ASR 모델](https://huggingface.co/icig/announcer-korean-ipa-translation)


## 아키택쳐

- Backend: FastAPI
- DB: Sqlite
- Deployment: Docker


## 프로젝트 구조

```text
.
├── api/v1/
│   ├── endpoints/audio.py
│   └── api.py
├── core/
│   ├── config.py
│   └── logger.py
├── ipa/
│   └── ...
├── middleware/
│   └── ...
├── session/
│   └── ...
├── Dockerfile
├── main.py
├── requirements.txt
└── schema.py
```

- `api/v1/`: 
  RestAPI를 구현하는 패키지
  - `endpoints/audio.py`: 음성인식 관련 api를 구현
  - `api.py`: api router로 구현한 함수를 api 경로와 매칭
- `core/`: FastAPI 핵심 설정
  - `config.py`: FastAPI의 기본적인 설정
  - `logger.py`: api 호출-응답 로거
- `ipa/`: IPA 변환 api를 위한 패키지
- `middleware/`: 안증/로깅을 위한 미들웨어
- `session/`: api에서 db/모델 사용을 위한 세션
- `Dockerfile`: Docker 배포를 위한 Dockerfile
- `main.py`: FastAPI main
- `requirements.txt`: 패키지 정보
- `schema.py`: api 요청/응답 스키마


## 환경 & 의존성

```text
python>=3.11
fastapi==0.112.2
pydantic==2.10.3
pydantic-settings==2.9.1
python-dotenv==1.1.0
gunicorn==23.0.0
uvicorn==0.32.1
sqlalchemy==2.0.40
pymysql==1.1.1
transformers==4.51.3
huggingface==0.0.1
torch==2.7.0
torchaudio==2.7.0
torchvision==0.22.0
python-multipart==0.0.20
soundfile==0.13.1
librosa==0.11.0
httpx==0.28.1
aiosqlite==0.21.0
```


## 기여

- 김준철 - 100%
  - 모든 작업 진행
