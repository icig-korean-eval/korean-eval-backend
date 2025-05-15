from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)

from core.config import settings


_model_asr: WhisperForConditionalGeneration | None = None
_model_ipa_announcer: Wav2Vec2ForCTC | None = None
_model_ipa_normal: Wav2Vec2ForCTC | None = None

_preprocessor_asr: WhisperProcessor | None = None
_preprocessor_ipa_announcer: Wav2Vec2Processor | None = None
_preprocessor_ipa_normal: Wav2Vec2Processor | None = None

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
        _model_ipa_announcer = Wav2Vec2ForCTC.from_pretrained(
            'icig/announcer-korean-ipa-translation',
            token=settings.HUGGINGFACE_KEY
        )
        _preprocessor_ipa_announcer = Wav2Vec2Processor.from_pretrained(
            'icig/announcer-korean-ipa-translation',
            token=settings.HUGGINGFACE_KEY
        )
        _model_ipa_normal = Wav2Vec2ForCTC.from_pretrained(
            'icig/normal-korean-ipa-translation',
            token=settings.HUGGINGFACE_KEY
        )
        _preprocessor_ipa_normal = Wav2Vec2Processor.from_pretrained(
            'icig/normal-korean-ipa-translation',
            token=settings.HUGGINGFACE_KEY
        )

def get_asr_model() -> tuple[
    WhisperForConditionalGeneration | WhisperForConditionalGeneration | None,
    WhisperProcessor | None
]:
    return _model_asr, _preprocessor_asr

def get_ipa_announcer_model():
    return _model_ipa_announcer, _preprocessor_ipa_announcer

def get_ipa_normal_model():
    return _model_ipa_normal, _preprocessor_ipa_normal
