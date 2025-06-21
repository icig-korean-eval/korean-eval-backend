from pydantic import BaseModel, Field
from typing import List, Optional, Union, Any, Dict
from datetime import date, datetime



# 음성 인식 결과 모델
class Transcribe(BaseModel):
    transcription: str


# IPA(국제 음성 기호) 변환 결과 모델
class Ipa(BaseModel):
    ipa: str


# IPA 변환 요청 시 사용되는 입력 모델
class IpaPost(BaseModel):
    text: str


# 자모 및 음절 단위 전사 정보를 담는 모델
class Syllables(BaseModel):
    jamo: List[str]
    transcript: List[str]

# 단어에 대한 정보를 담는 모델
class WordData(BaseModel):
    value: str
    syllables: Syllables

# 전체 IPA 변환 결과를 구조화해서 담는 모델
class IpaResult(BaseModel):
    original: str
    words: Dict[int, WordData]
    result: str
    result_array: List[str]

# 채팅 입력 요청 모델 (기존 채팅방에 메시지 추가)
class ChatPost(BaseModel):
    chat_id: str
    message: str

# 채팅방 생성 요청 모델
class ChatCreate(BaseModel):
    situation: str

# 채팅방 생성에 대한 응답 모델
class ChatCreateResponse(BaseModel):
    chat_id: str

# 채팅 응답에 대한 모델 (모델의 답변과 피드백 포함)
class ChatPostResponse(BaseModel):
    reply: str
    feedback: Any


# 문법 오류 정보를 담는 모델
class GrammaticalError(BaseModel):
    incorrect_part: str = Field(alias="Incorrect part")
    corrected_version: str = Field(alias="Corrected version")
    reason: str = Field(alias="Reason")


# 더 나은 표현 제안을 담는 모델
class BetterExpression(BaseModel):
    original_part: str = Field(alias="Original part")
    suggestion: str = Field(alias="Suggestion")
    reason: str = Field(alias="Reason")


# 하나의 메시지에 대한 종합 피드백 모델
class Feedback(BaseModel):
    grammatical_errors: List[GrammaticalError]
    better_expressions: List[BetterExpression]

# 채팅 응답과 그에 대한 피드백을 함께 담는 모델
class ConversationFeedback(BaseModel):
    reply: str
    feedback: Feedback


# 채팅방의 메타 정보를 담는 모델
class ChatRoom(BaseModel):
    chat_id: str
    title: str
    situation: str
    created_at: datetime


# 채팅 메시지를 표현하는 모델
class ChatMessage(BaseModel):
    id: int
    chat_id: str
    timestamp: datetime
    role: str
    content: str
