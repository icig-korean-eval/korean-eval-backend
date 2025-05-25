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
from typing import Literal

from ipa.src.worker import convert

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
    result = convert(request.text, rules_to_apply='pastcnovr')
    print(result)

    return result
