from pydantic import BaseModel, Field
from typing import List, Optional, Union, Any, Dict
from datetime import date, datetime


class Transcribe(BaseModel):
    transcription: str


class Ipa(BaseModel):
    ipa: str


class IpaPost(BaseModel):
    text: str


class Syllables(BaseModel):
    jamo: List[str]
    transcript: List[str]

class WordData(BaseModel):
    value: str
    syllables: Syllables

class IpaResult(BaseModel):
    original: str
    words: Dict[int, WordData]
    result: str
    result_array: List[str]

class ChatPost(BaseModel):
    chat_id: str
    message: str

class ChatCreate(BaseModel):
    situation: str

class ChatCreateResponse(BaseModel):
    chat_id: str

class ChatPostResponse(BaseModel):
    reply: str
    feedback: Any
