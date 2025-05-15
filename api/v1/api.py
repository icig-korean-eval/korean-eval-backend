from fastapi import APIRouter

from api.v1.endpoints import (
    example,
    audio
)


router = APIRouter()
router.include_router(example.router)
router.include_router(audio.router)
