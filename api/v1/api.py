from fastapi import APIRouter

from api.v1.endpoints import (
    example
)


router = APIRouter()
router.include_router(example.router)
