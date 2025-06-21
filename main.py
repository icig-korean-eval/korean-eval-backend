from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from api.v1.api import router as router_v1
from core.config import settings
from core.logger import logger

from session.model_session import load_model

import sqlite3

from middleware.logging_middleware import LoggingMiddleware
from middleware.auth_middleware import AuthMiddleware
from fastapi.middleware.cors import CORSMiddleware


# 예외 처리, 미들웨어, 라우터 등록, DB 테이블 생성 등 FastAPI 앱 초기화 코드
async def default_error_handler(_: Request, exception: Exception) -> JSONResponse:
    logger.exception(exception)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={'message': 'Unhandled Internal Server Error'}
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션의 생명 주기(lifespan)를 관리하는 함수.
    앱 시작 전 초기화 작업과 종료 시 클린업 작업을 정의할 수 있음.
    """
    # 모델 로딩
    load_model()

    # SQLite 연결 및 테이블 초기화
    conn = sqlite3.connect(settings.DB_PATH)
    c = conn.cursor()

    # 채팅 기록 테이블 생성 (존재하지 않을 경우만)
    c.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        role TEXT,
        content TEXT
    )
    """)

    # 채팅방 정보 테이블 생성 (존재하지 않을 경우만)
    c.execute("""
    CREATE TABLE IF NOT EXISTS chat_room (
        chat_id TEXT PRIMARY KEY,
        title TEXT,
        situation TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()  # 변경 사항 저장
    conn.close()  # DB 연결 종료

    logger.info("Application startup completed.")
    # logger.info(settings.FASTAPI_ENV)
    # logger.debug(logger.handlers)

    # 비동기 컨텍스트 매니저 시작
    yield
    logger.info("Application shutdown.")


# FastAPI 앱 생성 시 lifespan 컨텍스트를 함께 지정
app = FastAPI(lifespan=lifespan)

# CORS(Cross-Origin Resource Sharing) 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        '*'
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 사용자 정의 인증 미들웨어 추가
app.add_middleware(AuthMiddleware)
# 요청/응답 로깅 미들웨어 추가
app.add_middleware(LoggingMiddleware)

# 모든 예외에 대해 위에서 정의한 기본 에러 핸들러 사용
app.add_exception_handler(Exception, default_error_handler)


# v1 API 라우터 등록, 설정된 prefix로 경로 지정 (예: /api/v1)
app.include_router(router_v1, prefix=settings.API_V1_STR)
