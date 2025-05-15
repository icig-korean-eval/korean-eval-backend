from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)

from core.config import settings


_model_asr = None
_model_ipa_announcer = None
_model_ipa_normal = None

_preprocessor_asr = None
_preprocessor_ipa_announcer = None
_preprocessor_ipa_normal = None

def load_model():
    global _model_asr, _preprocessor_asr,\
        _model_ipa_announcer, _preprocessor_ipa_announcer,\
        _model_ipa_normal, _preprocessor_ipa_normal
    if _model_asr is None:
        _model_asr = WhisperForConditionalGeneration.from_pretrained(
            'icig/non-native-korean-speech-asr',
            token=settings.HUGGINGFACE_KEY
        )
        _preprocessor_asr = WhisperProcessor.from_pretrained(
            'openai/whisper-base',
            token=settings.HUGGINGFACE_KEY,
            language="Korean",
            task="transcribe"
        )
        _model_ipa_announcer = Wav2Vec2Processor.from_pretrained(
            'icig/announcer-korean-ipa-translation',
            token=settings.HUGGINGFACE_KEY
        )
        _preprocessor_ipa_announcer = Wav2Vec2ForCTC.from_pretrained(
            'icig/announcer-korean-ipa-translation',
            token=settings.HUGGINGFACE_KEY
        )
        _model_ipa_normal = Wav2Vec2Processor.from_pretrained(
            'icig/normal-korean-ipa-translation',
            token=settings.HUGGINGFACE_KEY
        )
        _preprocessor_ipa_normal = Wav2Vec2ForCTC.from_pretrained(
            'icig/normal-korean-ipa-translation',
            token=settings.HUGGINGFACE_KEY
        )

def get_asr_model():
    return _model_asr, _preprocessor_asr

def get_ipa_announcer_model():
    return _model_ipa_announcer, _preprocessor_ipa_announcer

def get_ipa_normal_model():
    return _model_ipa_normal, _preprocessor_ipa_normal
