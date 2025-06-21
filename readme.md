# REST API Server for Korean Language Learning Program for Foreigners


- REST API server to provide models to users
- Serving fine-tuned models:
  - [Non-native Korean Speech ASR Model](https://huggingface.co/icig/non-native-korean-speech-asr)
  - [Standard Korean IPA ASR Model](https://huggingface.co/icig/normal-korean-ipa-translation)
  - [Announcer Korean IPA ASR Model](https://huggingface.co/icig/announcer-korean-ipa-translation)

  
## Architecture

- Backend: FastAPI  
- Database: SQLite  
- Deployment: Docker


## Project Structure

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

- `api/v1/`: Package implementing the REST API
  - `endpoints/audio.py`: Implements APIs related to speech recognition
  - `api.py`: Maps API functions to API routes using routers
- `core/`: Core configuration for FastAPI
  - `config.py`: Basic settings for FastAPI
  - `logger.py`: Logs API request and response events
- `ipa/`: Package for IPA conversion API
- `middleware/`: Middleware for authentication and logging
- `session/`: Session management for database/model access in API
- `Dockerfile`: Dockerfile for containerized deployment
- `main.py`: FastAPI application entry point
- `requirements.txt`: List of required packages
- `schema.py`: API request/response schemas


## Environment & Dependencies

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


## Contribution

- Joonchul Kim - 100%  
  - Completed all development and implementation
