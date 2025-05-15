from starlette.middleware.base import BaseHTTPMiddleware
from core.logger import logger, request_id_context
import uuid
import time


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        request_id = str(uuid.uuid4())
        request_id_context.set(request_id)

        logger.info(f"Request: {request.method} {request.url}")

        response = await call_next(request)

        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"Response: {request.method} {request.url} | Status: {response.status_code} | Time: {process_time:.2f}ms"
        )
        return response
