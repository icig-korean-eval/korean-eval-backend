from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from api.v1.api import router as router_v1
from core.config import settings
from core.logger import logger

from session.model_session import load_model

from middleware.logging_middleware import LoggingMiddleware
from middleware.auth_middleware import AuthMiddleware
from fastapi.middleware.cors import CORSMiddleware


async def default_error_handler(_: Request, exception: Exception) -> JSONResponse:
    logger.exception(exception)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={'message': 'Unhandled Internal Server Error'}
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    logger.info("Application startup completed.")
    # logger.info(settings.FASTAPI_ENV)
    # logger.debug(logger.handlers)
    yield
    logger.info("Application shutdown.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        '*'
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)
app.add_middleware(LoggingMiddleware)

app.add_exception_handler(Exception, default_error_handler)

app.include_router(router_v1, prefix=settings.API_V1_STR)
