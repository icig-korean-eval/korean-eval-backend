from pydantic import BaseModel, Field
from typing import List, Optional, Union, Any
from datetime import date, datetime


class Transcribe(BaseModel):
    transcription: str


class Ipa(BaseModel):
    ipa: str
