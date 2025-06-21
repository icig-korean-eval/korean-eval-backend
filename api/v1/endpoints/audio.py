from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

import torch
from session.model_session import (
    get_asr_model,
    get_ipa_normal_model,
    get_ipa_announcer_model
)
import schema

import io
import soundfile as sf
import numpy as np
import librosa
from typing import Literal, List

from ipa.src.worker import convert

import httpx
import aiosqlite

from uuid import uuid4
import re

from core.config import settings

import json

# FastAPI 라우터 생성 (API 엔드포인트들을 묶는 단위)
router = APIRouter()


# 업로드된 오디오 파일을 읽고, 필요한 전처리를 수행하는 유틸리티 함수
async def read_audio(file):
    # 업로드된 파일을 비동기로 읽음 (bytes)
    audio_bytes = await file.read()
    try:
        # bytes 데이터를 오디오로 변환 (샘플링레이트 및 waveform 추출)
        audio_input, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception as e:
        # 파일이 오디오 형식이 아닐 경우 예외 처리
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    # 다채널(스테레오)일 경우 평균을 내어 모노로 변환
    if audio_input.ndim > 1:
        audio_input = np.mean(audio_input, axis=1)

    # 샘플링 레이트가 16000Hz가 아니면 리샘플링
    if sr != 16000:
        audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=16000)
        sr = 16000

    # 전처리된 오디오와 샘플링 레이트 반환
    return audio_input, sr


# [POST] /transcribe - Whisper 모델을 이용한 음성 → 텍스트 전사 API
@router.post("/transcribe", response_model=schema.Transcribe)
async def transcribe(
    file: UploadFile = File(..., media_type='audio/*'), # audio/* 형식의 파일 업로드 필수
    asr = Depends(get_asr_model), # 의존성 주입을 통해 모델과 전처리기를 가져옴
):
    model_asr, preprocessor_asr = asr
    # 모델이나 전처리기가 로드되지 않은 경우 에러 반환
    if model_asr is None or preprocessor_asr is None:
        raise HTTPException(status_code=503, detail="Model or processor not loaded")

    # 오디오 파일 읽고 전처리 수행
    audio_input, sr = await read_audio(file)

    # Whisper 전처리기 입력 형식으로 변환
    input_features = preprocessor_asr(
        audio_input,
        sampling_rate=sr,
        return_tensors="pt",
    ).input_features

    # 모델 예측 수행
    with torch.no_grad():
        predicted_ids = model_asr.generate(input_features)[0]
    # ID를 텍스트로 디코딩 후 공백 제거
    transcription = preprocessor_asr.decode(predicted_ids).strip()

    return {"transcription": transcription}


# [POST] /ipa - 음성을 IPA(국제 음성 기호)로 변환하는 API
@router.post("/ipa", response_model=schema.Ipa)
async def convert_ipa(
    type: Literal["normal", "announcer"] = "normal",  # 요청에서 변환 유형(normal 또는 announcer) 지정
    file: UploadFile = File(..., media_type='audio/*'),  # 오디오 파일 업로드 필수
    announcer=Depends(get_ipa_announcer_model),  # 아나운서 모델 의존성 주입
    normal=Depends(get_ipa_normal_model),  # 일반 모델 의존성 주입
):
    # 선택한 타입에 따라 적절한 모델 선택
    if type == 'normal':
        model_ipa, preprocessor_ipa = normal
    else:
        model_ipa, preprocessor_ipa = announcer
    # 모델 또는 전처리기가 로드되지 않은 경우 오류 반환
    if model_ipa is None or preprocessor_ipa is None:
        raise HTTPException(status_code=503, detail="Model or processor not loaded")

    # 오디오 파일 읽고 전처리
    audio_input, sr = await read_audio(file)

    # Wav2Vec2 전처리기 입력값 생성
    input_values = preprocessor_ipa.feature_extractor(
        audio_input, sampling_rate=sr
    ).input_values[0]
    input_values = torch.tensor(input_values, dtype=torch.float)

    # 예측 수행
    with torch.no_grad():
        logits = model_ipa(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)

    # 결과 ID를 텍스트(IPA)로 디코딩
    pred_text = preprocessor_ipa.tokenizer.batch_decode(pred_ids, skip_special_tokens=True).strip()

    return {"ipa": pred_text}


# [POST] /ipa/text - 텍스트를 IPA로 변환하는 API
@router.post("/ipa/text", response_model=schema.IpaResult)
async def convert_ipa(
    request: schema.IpaPost # 요청 body에 포함된 텍스트
):
    # convert 함수는 사전 정의된 변환 규칙 'pastcnovr'를 적용하여 IPA 변환을 수행
    return convert(request.text, rules_to_apply='pastcnovr')


# Ollama에게 대화 이력을 전달하여 응답을 받는 함수
async def query_ollama(history: list[dict]):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://175.212.190.95:8081/api/chat",
            headers={
                "Authorization": f"Bearer {settings.OLLAMA_KEY}"
            },
            json={
                "model": "gemma3:27b",
                "messages": history,
                "stream": False
            }
        )

        # 응답 상태가 200이 아닐 경우 오류 출력
        if response.status_code != 200:
            print("Ollama API 호출 실패:")
            print(f"Status code: {response.status_code}")
            print(f"Response body: {response.text}")
        return response.json()


# Ollama에게 문법 및 표현 피드백 요청
async def query_ollama_feedback(message: str, situation: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://175.212.190.95:8081/api/generate",
            headers={
                "Authorization": f"Bearer {settings.OLLAMA_KEY}"
            },
            json={
                "model": "gemma3:27b",
                "prompt": f'You are a native Korean teaching Korean to foreigners. The user is a Korean learner. Given the situation: \'{situation}\' and the user\'s sentence: \'{message}\', return a JSON object with exactly two keys: "grammatical_errors" and "better_expressions". Each must be a list of 0–3 strings. Grammar items must state the incorrect part, correction, and reason. Expression items must show the original and a better alternative. Output only valid plain JSON with no markdown, no explanation, and absolutely no code block or ```json tags. The output must be directly parsable using Python\'s json.loads(). Example: {{"grammatical_errors": [{{"Incorrect part": "갔습니다","Corrected version": "갔어요". "Reason": "\'갔습니다\' is too formal in this context."}}], "better_expressions": [{{"Original part": "기분이 나쁘지 않아요", "Suggestion": "기분이 좋아요", "Reason": "More natural and concise."}}]}}',
                "stream": False
            }
        )
        if response.status_code != 200:
            print("Ollama API 호출 실패:")
        print(f"Status code: {response.status_code}")
        print(f"Response body: {response.text}")
        return response.json()


# Ollama에게 상황 설명(situation)에 기반한 제목(title) 생성 요청
async def query_ollama_title(situation: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://175.212.190.95:8081/api/generate",
            headers={
                "Authorization": f"Bearer {settings.OLLAMA_KEY}"
            },
            json={
                "model": "gemma3:27b",
                "prompt": f'Given a description of a situation, generate a short and meaningful chat room title that summarizes the situation in 20 characters or fewer (including spaces and punctuation). Only output the title sentence. Do not include any explanation, formatting, or additional text. The title must always be in English. Situation: {situation}',
                "stream": False
            }
        )
        if response.status_code != 200:
            print("Ollama API 호출 실패:")
        print(f"Status code: {response.status_code}")
        print(f"Response body: {response.text}")
        return response.json()


# 특정 chat_id에 해당하는 전체 대화 메시지를 가져오는 함수
async def get_chat_history(chat_id: str) -> List[dict]:
    messages = []
    async with aiosqlite.connect(settings.DB_PATH) as db:
        async with db.execute("SELECT role, content FROM chat_history WHERE chat_id = ? ORDER BY timestamp", (chat_id,)) as cursor:
            async for row in cursor:
                messages.append({"role": row[0], "content": row[1]})
    return messages


# 특정 chat_id에 대한 상황(situation) 값을 가져오는 함수
async def get_chat_situation(chat_id: str) -> str:
    async with aiosqlite.connect(settings.DB_PATH) as db:
        async with db.execute("SELECT situation FROM chat_room WHERE chat_id = ?", (chat_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return row[0]
            else:
                return ""

# 새 메시지를 DB에 저장하는 함수
async def save_message(chat_id: str, role: str, content: str):
    async with aiosqlite.connect(settings.DB_PATH) as db:
        await db.execute(
            "INSERT INTO chat_history (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, role, content)
        )
        await db.commit()


# [POST] /chat : 새로운 채팅방을 생성하는 API
@router.post('/chat', response_model=schema.ChatRoom)
async def chat_create(
    request: schema.ChatCreate # 상황(situation) 필드를 포함한 요청 바디
):
    situation = request.situation
    chat_id = str(uuid4()) # 고유 chat_id 생성

    # Ollama에게 전달할 초기 system 프롬프트 생성
    description = f'You are a native Korean helping a non-native user practice Korean conversation based on the situation: \'{situation}\'. Speak only in Korean and always guide the user to continue the dialogue naturally. After each user message, respond your reply in Korean. Use plain text only, no markdown or symbols.'

    # 상황 설명으로부터 제목 생성 요청
    title = await query_ollama_title(situation)
    title = title['response']

    # 1. chat_room에 insert
    async with aiosqlite.connect(settings.DB_PATH) as db:
        # 중복 확인
        async with db.execute("SELECT 1 FROM chat_history WHERE chat_id = ?", (chat_id,)) as cursor:
            exists = await cursor.fetchone()
            if exists:
                raise HTTPException(status_code=400, detail="Chat ID already exists")

        # 채팅방 정보 삽입
        await db.execute(
            "INSERT INTO chat_room (chat_id, title, situation) VALUES (?, ?, ?)",
            (chat_id, title, situation)
        )
        # system 메시지 삽입 (초기 지시 프롬프트)
        await db.execute(
            "INSERT INTO chat_history (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, "system", description)
        )
        await db.commit()

        # 생성한 채팅방 정보를 다시 조회
        async with db.execute(
                "SELECT * FROM chat_room WHERE chat_id = ?",
                (chat_id,)
        ) as cursor:
            row = await cursor.fetchone()

    if not row:
        # 이론상 발생 안 하지만, 혹시를 대비해
        raise HTTPException(status_code=500, detail="Failed to retrieve created chat room")
    return schema.ChatRoom(**{
        'chat_id': row[0],
        'title': row[1],
        'situation': row[2],
        'created_at': row[3],
    })


# 문자열에서 "응답:" 이후의 텍스트만 추출하는 유틸리티 함수
def extract_response(text: str) -> str:
    match = re.search(r"응답:\s*(.*?)(?:\n|$)", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# [POST] /chat/conversation : 사용자 메시지에 응답 + 피드백 제공
@router.post('/chat/conversation', response_model=schema.ConversationFeedback)
async def chat_conversation(
    request: schema.ChatPost # 요청은 chat_id와 message 필드를 포함
):
    chat_id = request.chat_id
    user_msg = request.message

    # 현재 채팅방의 상황 정보 조회
    situation = await get_chat_situation(chat_id)

    # 현재 채팅방의 대화 이력 가져오기
    history = await get_chat_history(chat_id)
    # 새 유저 메시지를 대화 이력에 추가
    history.append({"role": "user", "content": user_msg})

    try:
        # Ollama에 전체 이력을 보내 응답 생성
        result = await query_ollama(history)
        assistant_reply = result["message"]["content"]

        print(1, assistant_reply)

        # 유저 메시지에 대한 피드백 요청 (문법/표현 오류)
        feedback_reply = await query_ollama_feedback(user_msg, situation)
        feedback_reply = feedback_reply['response']

        # 코드블럭 마크다운 제거 (```json ... ```)
        if feedback_reply.startswith("```json"):
            feedback_reply = feedback_reply[len("```json"):].strip()
        if feedback_reply.endswith("```"):
            feedback_reply = feedback_reply[:-3].strip()

        print(2, feedback_reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 대화 이력을 DB에 저장 (유저 → 어시스턴트 순)
    await save_message(chat_id, "user", user_msg)
    await save_message(chat_id, "assistant", assistant_reply)

    return {
        "reply": assistant_reply,
        'feedback': json.loads(feedback_reply)
    }


# [GET] /chat/rooms : 모든 채팅방 목록 조회
@router.get("/chat/rooms", response_model=List[schema.ChatRoom])
async def get_all_chat_rooms():
    async with aiosqlite.connect(settings.DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute("SELECT * FROM chat_room ORDER BY created_at DESC") as cursor:
            rows = await cursor.fetchall()

        return [schema.ChatRoom(**dict(row)) for row in rows]


# [GET] /chat/{chat_id} : 특정 채팅방 정보 조회
@router.get("/chat/{chat_id}", response_model=schema.ChatRoom)
async def get_chat_room_info(chat_id: str):
    async with aiosqlite.connect(settings.DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute(
            "SELECT * FROM chat_room WHERE chat_id = ?",
            (chat_id,)
        ) as cursor:
            row = await cursor.fetchone()

    if row:
        return schema.ChatRoom(**{
            'chat_id': row[0],
            'title': row[1],
            'situation': row[2],
            'created_at': row[3],
        })
    return None


# [DELETE] /chat/{chat_id} : 채팅방 및 해당 대화 삭제
@router.delete("/chat/{chat_id}")
async def remove_chat_room(chat_id: str):
    async with aiosqlite.connect(settings.DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 1) 먼저 삭제할 행을 조회
        async with db.execute(
            "SELECT * FROM chat_room WHERE chat_id = ?",
            (chat_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            # 해당 chat_id가 없으면 404
            raise HTTPException(status_code=404, detail="Chat room not found")

        # 2) 실제로 DELETE 실행
        await db.execute(
            "DELETE FROM chat_history WHERE chat_id = ?",
            (chat_id,)
        )

        await db.execute(
            "DELETE FROM chat_room WHERE chat_id = ?",
            (chat_id,)
        )
        await db.commit()


# [GET] /chat/{chat_id}/history : 특정 채팅방의 대화 이력 조회
@router.get("/chat/{chat_id}/history", response_model=List[schema.ChatMessage])
async def get_chat_history_api(chat_id: str):
    async with aiosqlite.connect(settings.DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        query = """
                    SELECT * FROM chat_history
                    WHERE chat_id = ?
                      AND role != 'system'
                    ORDER BY timestamp ASC, 
                             CASE role
                                 WHEN 'user' THEN 0
                                 ELSE 1
                             END
                """
        async with db.execute(
            query,
            (chat_id,)
        ) as cursor:
            rows = await cursor.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="Chat history not found")

        return [schema.ChatMessage(**dict(row)) for row in rows]
