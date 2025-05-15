from fastapi import APIRouter, status, Request, HTTPException, Depends
from fastapi.responses import Response

from sqlalchemy.orm import Session

router = APIRouter()

@router.get('/example')
async def say_hello(name: str):
    return {"message": f"Hello {name}"}