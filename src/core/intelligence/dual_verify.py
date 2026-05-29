"""Dual-model verification for images and voice transcriptions."""

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    """Результат двойной верификации."""

    answer: str
    model_a: str
    model_b: str
    model_arbiter: str | None = None
    agreement: bool = True
    confidence: float = 1.0


async def verify_image(
    image_data: bytes,
    image_mime: str,
    provider_a,  # VisionProvider
    provider_b,  # VisionProvider
    arbiter_provider=None,  # VisionProvider | None
    prompt: str = "Опиши что на изображении.",
    disagreement_threshold: float = 0.3,
) -> VerifyResult:
    """Анализирует изображение двумя провайдерами. При расхождении — арбитр."""
    # Запускаем двух провайдеров параллельно с таймаутом
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                provider_a.chat_with_image(image_data, image_mime, prompt),
                provider_b.chat_with_image(image_data, image_mime, prompt),
                return_exceptions=True,
            ),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        return VerifyResult(
            answer="",
            model_a=provider_a._model,
            model_b=provider_b._model,
            agreement=False,
            confidence=0.0,
        )

    answer_a = results[0].description if not isinstance(results[0], Exception) else ""
    answer_b = results[1].description if not isinstance(results[1], Exception) else ""

    # Простое сравнение: если оба непустые и похожи по длине — согласны
    if answer_a and answer_b:
        len_a, len_b = len(answer_a), len(answer_b)
        similarity = (
            min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 0
        )
        agreement = similarity > (1 - disagreement_threshold)
    else:
        agreement = False
        similarity = 0.0

    if not agreement and arbiter_provider:
        arbiter_prompt = f"Модель A ответила:\n{answer_a[:500]}\n\nМодель B ответила:\n{answer_b[:500]}\n\nДай свой вердикт: что на изображении?"
        arbiter_answer = await arbiter_provider.chat_with_image(
            image_data, image_mime, arbiter_prompt
        )
        return VerifyResult(
            answer=arbiter_answer,
            model_a=provider_a._model,
            model_b=provider_b._model,
            model_arbiter=arbiter_provider._model,
            agreement=False,
            confidence=0.5,
        )

    return VerifyResult(
        answer=answer_a or answer_b,
        model_a=provider_a._model,
        model_b=provider_b._model,
        agreement=agreement,
        confidence=similarity if agreement else 0.5,
    )


async def verify_transcription(
    audio_path,  # Path
    stt_a,  # TranscriptionService-compatible
    stt_b,  # TranscriptionService-compatible
    arbiter_stt=None,
    disagreement_threshold: float = 0.3,
) -> VerifyResult:
    """Расшифровывает аудио двумя STT-провайдерами. При расхождении — арбитр."""
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                stt_a.transcribe(audio_path),
                stt_b.transcribe(audio_path),
                return_exceptions=True,
            ),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        return VerifyResult(
            answer="",
            model_a=getattr(stt_a, "_model", "stt_a"),
            model_b=getattr(stt_b, "_model", "stt_b"),
            agreement=False,
            confidence=0.0,
        )

    text_a = results[0] if not isinstance(results[0], Exception) else ""
    text_b = results[1] if not isinstance(results[1], Exception) else ""

    if text_a and text_b:
        # Простое сравнение по словам
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if words_a and words_b:
            overlap = len(words_a & words_b) / len(words_a | words_b)
            agreement = overlap > (1 - disagreement_threshold)
        else:
            agreement = False
            overlap = 0.0
    else:
        agreement = False
        overlap = 0.0

    if not agreement and arbiter_stt:
        arbiter_text = await arbiter_stt.transcribe(audio_path)
        return VerifyResult(
            answer=arbiter_text,
            model_a=getattr(stt_a, "_model", "stt_a"),
            model_b=getattr(stt_b, "_model", "stt_b"),
            model_arbiter=getattr(arbiter_stt, "_model", "arbiter"),
            agreement=False,
            confidence=0.5,
        )

    return VerifyResult(
        answer=text_a or text_b,
        model_a=getattr(stt_a, "_model", "stt_a"),
        model_b=getattr(stt_b, "_model", "stt_b"),
        agreement=agreement,
        confidence=overlap if agreement else 0.5,
    )
