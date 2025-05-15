from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from core.config import settings

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ['/openapi.json', '/docs']:
            return await call_next(request)

        # Authorization 헤더 가져오기
        auth_header: str = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized: Missing or invalid Authorization header"})

        token = auth_header.split(" ", 1)[1]
        # 토큰 검증
        if token == settings.AUTH_KEY:
            return JSONResponse(status_code=403, content={"detail": "Forbidden: Invalid API key"})

        # 통과된 요청만 다음 단계로
        response = await call_next(request)
        return response
