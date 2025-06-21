import logging
from logging.handlers import RotatingFileHandler, SocketHandler

from contextvars import ContextVar

from core.config import settings
import json
import traceback

from datetime import datetime


# 각 요청에 고유한 ID를 부여하고, 로깅 시 이 값을 로그에 포함시키기 위한 ContextVar 선언
request_id_context: ContextVar[str] = ContextVar("request_id", default="-")

# 각 로그 레코드에 request_id를 삽입하는 필터 클래스 정의
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        try:
            record.request_id = request_id_context.get()
        except:
            record.request_id = 'none'
        return True


# JSON 형식의 로그 출력을 위한 커스텀 Formatter
class JSONFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        if datefmt:
            s = datetime.fromtimestamp(record.created).strftime(datefmt)
        else:
            s = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")
        return s

    def format(self, record):
        log_record = {
            "level": record.levelname,
            "request_id": record.request_id,
            "message": record.getMessage(),
            "logger": record.name,
            "time": self.formatTime(record),
            "module": record.module,
            "pathname": record.pathname,
            "filename": record.filename,
            "funcName": record.funcName,
            "lineno": record.lineno,
            # "env": settings.FASTAPI_ENV
        }
        if record.exc_info:
            log_record["exception"] = ''.join(traceback.format_exception(*record.exc_info))

        return json.dumps(log_record)


def setup_logger():
    logger = logging.getLogger("app_logger")

    if len(logger.handlers) > 0:
        return logger

    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(RequestIdFilter())
    console_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)

    file_handler = RotatingFileHandler(f"{settings.LOG_DIR}/app.log", maxBytes=5*1024*1024, backupCount=8)
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(RequestIdFilter())
    file_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # file_handler.setFormatter(file_format)
    file_handler.setFormatter(JSONFormatter())

    # tcp_handler = AsynchronousLogstashHandler('host.docker.internal', 5044, database_path=None)
    # tcp_handler.setLevel(logging.DEBUG)
    # tcp_handler.setFormatter(console_format)
    # tcp_handler.setFormatter(JSONFormatter())

    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # if settings.FASTAPI_ENV is not None:
        #     logger.addHandler(tcp_handler)

    return logger


logger = setup_logger()