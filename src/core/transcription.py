import asyncio
import logging
from pathlib import Path

from src.db.repo import cache_transcript, get_cached_transcript
from src.db.session import get_session


logger = logging.getLogger(__name__)


class TranscriptionService:
    """Локальный faster-whisper / OpenAI Whisper / Gemini / Mistral (multimodal) / hybrid."""

    def __init__(self, model_size: str = "small") -> None:
        self._model_size = model_size
        self._model = None
        self._lock = asyncio.Lock()

    async def _ensure_local_model(self) -> object:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                def _load() -> object:
                    return WhisperModel(self._model_size, device="auto", compute_type="auto")

                self._model = await asyncio.to_thread(_load)
        return self._model

    async def _transcribe_local(self, path: Path, language: str | None) -> str:
        model = await self._ensure_local_model()

        def _run() -> str:
            segments, _info = model.transcribe(str(path), language=language)
            return " ".join(seg.text.strip() for seg in segments).strip()

        return await asyncio.to_thread(_run)

    async def _transcribe_openai(self, path: Path, openai_key: str, language: str | None) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=openai_key)
        with path.open("rb") as f:
            resp = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
            )
        return resp.text

    async def _transcribe_gemini(self, path: Path, gemini_key: str, language: str | None) -> str:
        from google import genai

        client = genai.Client(api_key=gemini_key)

        def _run() -> str:
            audio_file = client.files.upload(file=str(path))
            prompt = "Transcribe this audio verbatim. Return only the transcription text, nothing else."
            if language:
                prompt += f" The audio language is {language}."
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, audio_file],
            )
            return resp.text or ""

        return await asyncio.to_thread(_run)

    async def _transcribe_mistral(self, path: Path, mistral_key: str, language: str | None) -> str:
        import httpx

        suffix = path.suffix.lstrip(".") or "ogg"
        mime = f"audio/{suffix}" if suffix != "oga" else "audio/ogg"

        data = {"model": "voxtral-mini-transcribe-latest"}
        if language:
            data["language"] = language

        async with httpx.AsyncClient(timeout=120.0) as client:
            with path.open("rb") as f:
                resp = await client.post(
                    "https://api.mistral.ai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {mistral_key}"},
                    data=data,
                    files={"file": (path.name, f, mime)},
                )
            resp.raise_for_status()
            return resp.json().get("text", "")

    async def transcribe(
        self,
        path: Path,
        *,
        file_id: str | None = None,
        mode: str = "hybrid",
        openai_key: str | None = None,
        gemini_key: str | None = None,
        mistral_key: str | None = None,
        api_provider: str = "openai",
        language: str | None = None,
    ) -> str:
        if file_id:
            async with get_session() as session:
                cached = await get_cached_transcript(session, file_id)
                if cached:
                    return cached

        text = ""
        if mode == "api":
            text = await self._call_api_transcribe(path, openai_key, gemini_key, mistral_key, api_provider, language)
        elif mode == "local":
            text = await self._transcribe_local(path, language)
        else:  # hybrid
            try:
                text = await self._transcribe_local(path, language)
            except Exception:
                logger.exception("Local transcription failed, falling back to API")
                text = await self._call_api_transcribe(path, openai_key, gemini_key, mistral_key, api_provider, language)

        if file_id and text:
            async with get_session() as session:
                await cache_transcript(session, file_id, text)
        return text

    async def _call_api_transcribe(
        self,
        path: Path,
        openai_key: str | None,
        gemini_key: str | None,
        mistral_key: str | None,
        api_provider: str,
        language: str | None,
    ) -> str:
        if api_provider == "gemini":
            if not gemini_key:
                raise ValueError("Gemini API key required")
            return await self._transcribe_gemini(path, gemini_key, language)
        elif api_provider == "mistral":
            if not mistral_key:
                raise ValueError("Mistral API key required")
            return await self._transcribe_mistral(path, mistral_key, language)
        else:
            if not openai_key:
                raise ValueError("OpenAI API key required")
            return await self._transcribe_openai(path, openai_key, language)


transcription_service = TranscriptionService()
