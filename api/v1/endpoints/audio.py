from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

import torch
from session.model_session import (
    get_asr_model
)

import io
import soundfile as sf
import numpy as np
import librosa

router = APIRouter()

@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., media_type='audio/*'),
    asr = Depends(get_asr_model),
):
    model_asr, preprocessor_asr = asr
    if model_asr is None or preprocessor_asr is None:
        raise HTTPException(status_code=503, detail="Model or processor not loaded")

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

    input_features = preprocessor_asr(
        audio_input,
        sampling_rate=sr,
        return_tensors="pt",
    ).input_features

    with torch.no_grad():
        predicted_ids = model_asr.generate(input_features)[0]
    transcription = preprocessor_asr.decode(predicted_ids).strip()

    return {"transcription": transcription}
