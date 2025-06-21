from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)

from core.config import settings


# 사용 모델 및 전처리기 전역 변수 선언
# 초기에는 모두 None이며, 최초 load_model() 호출 시 로딩됨
_model_asr: WhisperForConditionalGeneration | None = None              # Whisper 기반 음성 인식 모델
_model_ipa_announcer: Wav2Vec2ForCTC | None = None                     # 아나운서 스타일 음성 → IPA 변환 모델
_model_ipa_normal: Wav2Vec2ForCTC | None = None                        # 일반 화자 음성 → IPA 변환 모델

_preprocessor_asr: WhisperProcessor | None = None                     # Whisper용 전처리기
_preprocessor_ipa_announcer: Wav2Vec2Processor | None = None          # 아나운서 IPA용 전처리기
_preprocessor_ipa_normal: Wav2Vec2Processor | None = None             # 일반 IPA용 전처리기

# 모델을 한 번만 로드하여 전역 변수에 저장하는 함수
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

# Whisper 기반 ASR(음성 인식) 모델과 전처리기를 반환하는 함수
def get_asr_model() -> tuple[
    WhisperForConditionalGeneration | WhisperForConditionalGeneration | None,
    WhisperProcessor | None
]:
    return _model_asr, _preprocessor_asr

# 아나운서 화자 IPA 변환 모델과 전처리기를 반환하는 함수
def get_ipa_announcer_model():
    return _model_ipa_announcer, _preprocessor_ipa_announcer

# 일반 화자 IPA 변환 모델과 전처리기를 반환하는 함수
def get_ipa_normal_model():
    return _model_ipa_normal, _preprocessor_ipa_normal
