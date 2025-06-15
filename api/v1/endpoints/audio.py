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

router = APIRouter()


async def read_audio(file):
    audio_bytes = await file.read()
    try:
        audio_input, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    if audio_input.ndim > 1:
        audio_input = np.mean(audio_input, axis=1)
    if sr != 16000:
        audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio_input, sr


@router.post("/transcribe", response_model=schema.Transcribe)
async def transcribe(
    file: UploadFile = File(..., media_type='audio/*'),
    asr = Depends(get_asr_model),
):
    model_asr, preprocessor_asr = asr
    if model_asr is None or preprocessor_asr is None:
        raise HTTPException(status_code=503, detail="Model or processor not loaded")

    audio_input, sr = await read_audio(file)

    input_features = preprocessor_asr(
        audio_input,
        sampling_rate=sr,
        return_tensors="pt",
    ).input_features

    with torch.no_grad():
        predicted_ids = model_asr.generate(input_features)[0]
    transcription = preprocessor_asr.decode(predicted_ids).strip()

    return {"transcription": transcription}


@router.post("/ipa", response_model=schema.Ipa)
async def convert_ipa(
    type: Literal["normal", "announcer"] = "normal",
    file: UploadFile = File(..., media_type='audio/*'),
    announcer = Depends(get_ipa_announcer_model),
    normal = Depends(get_ipa_normal_model),
):
    if type == 'normal':
        model_ipa, preprocessor_ipa = normal
    else:
        model_ipa, preprocessor_ipa = announcer
    if model_ipa is None or preprocessor_ipa is None:
        raise HTTPException(status_code=503, detail="Model or processor not loaded")

    audio_input, sr = await read_audio(file)

    input_values = preprocessor_ipa.feature_extractor(
        audio_input, sampling_rate=sr
    ).input_values[0]
    input_values = torch.tensor(input_values, dtype=torch.float)

    with torch.no_grad():
        logits = model_ipa(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)

    pred_text = preprocessor_ipa.tokenizer.batch_decode(pred_ids, skip_special_tokens=True).strip()

    return {"ipa": pred_text}


@router.post("/ipa/text", response_model=schema.IpaResult)
async def convert_ipa(
    request: schema.IpaPost
):
    return convert(request.text, rules_to_apply='pastcnovr')


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
        if response.status_code != 200:
            print("Ollama API 호출 실패:")
            print(f"Status code: {response.status_code}")
            print(f"Response body: {response.text}")
        return response.json()


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


async def get_chat_history(chat_id: str) -> List[dict]:
    messages = []
    async with aiosqlite.connect(settings.DB_PATH) as db:
        async with db.execute("SELECT role, content FROM chat_history WHERE chat_id = ? ORDER BY timestamp", (chat_id,)) as cursor:
            async for row in cursor:
                messages.append({"role": row[0], "content": row[1]})
    return messages


async def get_chat_situation(chat_id: str) -> str:
    async with aiosqlite.connect(settings.DB_PATH) as db:
        async with db.execute("SELECT situation FROM chat_room WHERE chat_id = ?", (chat_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return row[0]
            else:
                return ""

async def save_message(chat_id: str, role: str, content: str):
    async with aiosqlite.connect(settings.DB_PATH) as db:
        await db.execute(
            "INSERT INTO chat_history (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, role, content)
        )
        await db.commit()


@router.post('/chat', response_model=schema.ChatRoom)
async def chat_create(
    request: schema.ChatCreate
):
    situation = request.situation
    chat_id = str(uuid4())

    description = f'You are a native Korean helping a non-native user practice Korean conversation based on the situation: \'{situation}\'. Speak only in Korean and always guide the user to continue the dialogue naturally. After each user message, respond your reply in Korean. Use plain text only, no markdown or symbols.'

    title = await query_ollama_title(situation)
    title = title['response']

    # 1. chat_room에 insert
    async with aiosqlite.connect(settings.DB_PATH) as db:
        # 중복 확인
        async with db.execute("SELECT 1 FROM chat_history WHERE chat_id = ?", (chat_id,)) as cursor:
            exists = await cursor.fetchone()
            if exists:
                raise HTTPException(status_code=400, detail="Chat ID already exists")

        await db.execute(
            "INSERT INTO chat_room (chat_id, title, situation) VALUES (?, ?, ?)",
            (chat_id, title, situation)
        )
        await db.execute(
            "INSERT INTO chat_history (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, "system", description)
        )
        await db.commit()

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


def extract_response(text: str) -> str:
    match = re.search(r"응답:\s*(.*?)(?:\n|$)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


@router.post('/chat/conversation', response_model=schema.ConversationFeedback)
async def chat_conversation(
    request: schema.ChatPost
):
    chat_id = request.chat_id
    user_msg = request.message

    situation = await get_chat_situation(chat_id)

    history = await get_chat_history(chat_id)
    history.append({"role": "user", "content": user_msg})

    try:
        result = await query_ollama(history)
        assistant_reply = result["message"]["content"]

        print(1, assistant_reply)

        feedback_reply = await query_ollama_feedback(user_msg, situation)
        feedback_reply = feedback_reply['response']

        if feedback_reply.startswith("```json"):
            feedback_reply = feedback_reply[len("```json"):].strip()
        if feedback_reply.endswith("```"):
            feedback_reply = feedback_reply[:-3].strip()

        print(2, feedback_reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    await save_message(chat_id, "user", user_msg)
    await save_message(chat_id, "assistant", assistant_reply)

    return {
        "reply": assistant_reply,
        'feedback': json.loads(feedback_reply)
    }


@router.get("/chat/rooms", response_model=List[schema.ChatRoom])
async def get_all_chat_rooms():
    async with aiosqlite.connect(settings.DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute("SELECT * FROM chat_room ORDER BY created_at DESC") as cursor:
            rows = await cursor.fetchall()

        return [schema.ChatRoom(**dict(row)) for row in rows]


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
