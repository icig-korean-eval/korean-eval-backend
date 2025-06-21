from starlette.middleware.base import BaseHTTPMiddleware
from core.logger import logger, request_id_context
import uuid
import time


# FastAPI에서 요청과 응답의 로그를 기록하는 미들웨어 정의
class LoggingMiddleware(BaseHTTPMiddleware):
    # 모든 요청에 대해 호출되는 메서드
    async def dispatch(self, request, call_next):
        # 요청 처리 시작 시각 기록 (성능 측정용)
        start_time = time.time()

        # 고유한 요청 ID 생성 (요청 추적에 활용)
        request_id = str(uuid.uuid4())
        # context variable에 저장 (추후 로그 추적용)
        request_id_context.set(request_id)

        # 요청 로그 출력 (HTTP 메서드, URL)
        logger.info(f"Request: {request.method} {request.url}")

        # 실제 요청을 다음 핸들러로 전달하여 처리
        response = await call_next(request)

        # 응답 처리 완료 후 시간 계산 (ms 단위)
        process_time = (time.time() - start_time) * 1000
        # 응답 로그 출력 (메서드, URL, 상태 코드, 처리 시간)
        logger.info(
            f"Response: {request.method} {request.url} | Status: {response.status_code} | Time: {process_time:.2f}ms"
        )
        return response
