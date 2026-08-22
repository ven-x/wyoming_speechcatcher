import asyncio
import logging
import sys
import numpy as np
from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from wyoming.asr import Transcribe, Transcript, TranscriptChunk, TranscriptStop
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler
from . import __version__
from .models import MODELS, TAGS

if TYPE_CHECKING:
    from .state import State

_LOGGER = logging.getLogger(__name__)

if sys.version_info >= (3, 9):
    _to_thread = asyncio.to_thread
else:

    async def _to_thread(func, *args, **kwargs):
        """Backport of asyncio.to_thread for Python 3.8."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

MODEL_RATE = 16000
MODEL_WIDTH = 2
MODEL_CHANNELS = 1
MIN_BUFFER_SAMPLES = 8000
FINAL_MIN_SAMPLES = 8000
MAX_BUFFER_SAMPLES = 30 * MODEL_RATE
LOCK_ACQUIRE_TIMEOUT = 30.0


def build_info(args: Any, state: "State") -> Info:
    return Info(
        asr=[
            AsrProgram(
                name="speechcatcher",
                description="EspNet2 streaming speech recognition",
                attribution=Attribution(
                    name="Benjamin Milde",
                    url="https://github.com/speechcatcher-asr/speechcatcher",
                ),
                installed=True,
                version=__version__,
                supports_transcript_streaming=getattr(
                    args, "stream_transcript", False
                ),
                requires_external_vad=getattr(args, "external_vad", False),
                models=[
                    AsrModel(
                        name=short_tag,
                        description=f"Speechcatcher {lang_code} ({short_tag})",
                        attribution=Attribution(
                            name="Benjamin Milde",
                            url=(
                                "https://github.com/speechcatcher-asr/"
                                "speechcatcher"
                            ),
                        ),
                        installed=state.model_available_locally(short_tag),
                        version=None,
                        languages=[lang_code],
                    )
                    for lang_code, short_tags in MODELS.items()
                    for short_tag in short_tags
                ],
            )
        ]
    )


class SpeechcatcherEventHandler(AsyncEventHandler):

    def __init__(
        self,
        args: Any,
        state: "State",
        reader: "asyncio.StreamReader",
        writer: "asyncio.StreamWriter",
    ) -> None:
        super().__init__(reader, writer)
        self.args = args
        self.state = state
        self.stream_transcript: bool = getattr(args, "stream_transcript", False)

        self._info_event = build_info(args, state).event()

        self.converter = AudioChunkConverter(
            rate=MODEL_RATE, width=MODEL_WIDTH, channels=MODEL_CHANNELS
        )

        self.language: Optional[str] = None
        self.model_name: Optional[str] = None
        self.speech2text: Optional[Any] = None
        self._model_load_failed: bool = False
        self._model_lock: Optional[asyncio.Lock] = None
        self._locked_model_name: Optional[str] = None
        self.audio_buffer = np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------
    async def handle_event(self, event: Event) -> bool:
        """Dispatch a Wyoming event. Returning False closes the connection."""
        if Describe.is_type(event.type):
            await self.write_event(self._info_event)
            _LOGGER.debug("Sent info")
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self.language = transcribe.language
            if transcribe.name is not None and transcribe.name not in TAGS:
                _LOGGER.warning(
                    "Unknown model name %r in transcribe event from client; "
                    "falling back to language default (language=%s)",
                    transcribe.name,
                    self.language,
                )
                self.model_name = None
            else:
                self.model_name = transcribe.name

            if (
                self._model_lock is not None
                and self._model_lock.locked()
                and self._locked_model_name is not None
            ):
                try:
                    new_resolved = self.state.resolve_model_name(
                        self.language, self.model_name
                    )
                except ValueError as exc:
                    _LOGGER.warning(
                        "transcribe while holding lock for %s could not "
                        "resolve new model (language=%s, model=%s): %s — "
                        "keeping current model lock (AUDIT-027-1 fallback)",
                        self._locked_model_name,
                        self.language,
                        self.model_name,
                        exc,
                    )
                    _LOGGER.debug(
                        "Transcribe requested: language=%s, model=%s",
                        self.language,
                        self.model_name,
                    )
                    return True

                if new_resolved != self._locked_model_name:
                    _LOGGER.info(
                        "transcribe changes model from %s to %s while "
                        "utterance lock is held — releasing old lock, "
                        "acquiring new (AUDIT-027-1 Variante a)",
                        self._locked_model_name,
                        new_resolved,
                    )
                    self._model_lock.release()
                    self._model_lock = None
                    old_locked = self._locked_model_name
                    self._locked_model_name = None
                    self.speech2text = None
                    self.audio_buffer = np.array([], dtype=np.float32)
                    await self._acquire_model_lock(new_resolved)
                    if self._model_load_failed:
                        _LOGGER.warning(
                            "Failed to acquire lock for new model %s "
                            "after releasing %s — degrading to empty-"
                            "transcript path (AUDIT-027-1 + 024-2)",
                            new_resolved,
                            old_locked,
                        )

            _LOGGER.debug(
                "Transcribe requested: language=%s, model=%s",
                self.language,
                self.model_name,
            )
            return True

        if AudioStart.is_type(event.type):
            if self._model_load_failed:
                _LOGGER.warning(
                    "audio-start after failed model/lock acquisition "
                    "(language=%s, model=%s) — ignoring (empty-transcript "
                    "path stays active)",
                    self.language,
                    self.model_name,
                )
                return True

            if self._model_lock is not None and self._model_lock.locked():
                _LOGGER.warning(
                    "Repeated audio-start on a connection that already holds "
                    "the model lock — treating as reset (no re-acquire)"
                )
                self.audio_buffer = np.array([], dtype=np.float32)
                if self.speech2text is not None:
                    self.speech2text.reset()
                return True

            self.audio_buffer = np.array([], dtype=np.float32)
            if self.speech2text is not None:
                self.speech2text.reset()

            try:
                resolved = self.state.resolve_model_name(
                    self.language, self.model_name
                )
            except ValueError as exc:
                _LOGGER.warning(
                    "audio-start could not resolve model (language=%s, "
                    "model=%s): %s — degrading to empty-transcript path",
                    self.language,
                    self.model_name,
                    exc,
                )
                self._model_load_failed = True
                return True
            await self._acquire_model_lock(resolved)
            return True

        if AudioChunk.is_type(event.type):
            await self._handle_audio_chunk(AudioChunk.from_event(event))
            return True

        if AudioStop.is_type(event.type):
            await self._handle_audio_stop()
            # One utterance per connection: close after the final transcript.
            return False

        _LOGGER.debug("Ignoring event type: %s", event.type)
        return True

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------
    async def _acquire_model_lock(self, resolved: str) -> None:
        lock = self.state.get_lock(resolved)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
        except asyncio.TimeoutError:
            _LOGGER.warning(
                "Timed out after %.0f s waiting for model lock %s — "
                "another connection is holding the utterance lock; "
                "degrading to empty-transcript path (AUDIT-024-2)",
                LOCK_ACQUIRE_TIMEOUT,
                resolved,
            )
            self._model_load_failed = True
            self._model_lock = None
            self._locked_model_name = None
            return
        self._model_lock = lock
        self._locked_model_name = resolved
        _LOGGER.debug("Model lock acquired for %s", resolved)

    async def _handle_audio_chunk(self, chunk: AudioChunk) -> None:
        if self._model_load_failed:
            return

        if self.speech2text is None:
            if self._model_lock is None:
                try:
                    resolved = self.state.resolve_model_name(
                        self.language, self.model_name
                    )
                except ValueError as exc:
                    _LOGGER.warning(
                        "audio-chunk could not resolve model (language=%s, "
                        "model=%s): %s — degrading to empty-transcript path",
                        self.language,
                        self.model_name,
                        exc,
                    )
                    self._model_load_failed = True
                    return
                await self._acquire_model_lock(resolved)
                if self._model_load_failed:
                    return
            try:
                # NOTE: get_model resolves (language, model_name) internally
                # with the same resolve_model_name() used for the lock, so
                # the locked model and the loaded model always match.
                self.speech2text = await _to_thread(
                    self.state.get_model, self.language, self.model_name
                )
            except (ValueError, RuntimeError) as exc:
                _LOGGER.error(
                    "Failed to acquire model (language=%s, model=%s): %s",
                    self.language,
                    self.model_name,
                    exc,
                )
                self._model_load_failed = True
                return
            _LOGGER.debug(
                "Acquired model for language=%s, model=%s",
                self.language,
                self.model_name,
            )

        chunk = self.converter.convert(chunk)

        audio_i16 = np.frombuffer(chunk.audio, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32767.0

        self.audio_buffer = np.concatenate([self.audio_buffer, audio_f32])
        if len(self.audio_buffer) > MAX_BUFFER_SAMPLES:
            _LOGGER.warning(
                "Audio buffer exceeded %d samples, dropping oldest audio",
                MAX_BUFFER_SAMPLES,
            )
            self.audio_buffer = self.audio_buffer[-MAX_BUFFER_SAMPLES:]

        if len(self.audio_buffer) < MIN_BUFFER_SAMPLES:
            return

        speech = self.audio_buffer
        self.audio_buffer = np.array([], dtype=np.float32)

        result = await _to_thread(self.speech2text, speech=speech, is_final=False)
        if not result:
            return

        if self.stream_transcript:
            text = result[0][0]
            if text:
                await self.write_event(TranscriptChunk(text=text).event())

    async def _handle_audio_stop(self) -> None:
        """Finalize the utterance, send the transcript and reset state.

        The final inference call runs in a worker thread via
        ``_to_thread`` (AUDIT-002). The model lock is released in a
        ``finally`` block (AUDIT-003, Variante B).
        """
        text = ""
        try:
            if self._model_load_failed:
                _LOGGER.warning(
                    "audio-stop after failed model/lock acquisition "
                    "(language=%s, model=%s) — sending empty transcript",
                    self.language,
                    self.model_name,
                )
            elif self.speech2text is not None:
                speech = self.audio_buffer
                self.audio_buffer = np.array([], dtype=np.float32)
                if 0 < len(speech) < FINAL_MIN_SAMPLES:
                    _LOGGER.debug(
                        "Padding final buffer from %d to %d samples with silence",
                        len(speech),
                        FINAL_MIN_SAMPLES,
                    )
                    speech = np.concatenate(
                        [speech, np.zeros(FINAL_MIN_SAMPLES - len(speech), dtype=np.float32)]
                    )
                elif len(speech) == 0:
                    _LOGGER.debug(
                        "audio-stop with empty buffer — sending %d samples of silence",
                        FINAL_MIN_SAMPLES,
                    )
                    speech = np.zeros(FINAL_MIN_SAMPLES, dtype=np.float32)

                call_kwargs: dict = {"speech": speech, "is_final": True}
                if getattr(self.args, "decoder", "native") == "native":
                    call_kwargs["finalize_all"] = True
                result = await _to_thread(self.speech2text, **call_kwargs)
                if result:
                    text = result[0][0] or ""
                else:
                    _LOGGER.warning("Model returned no result on audio-stop")

                self.speech2text.reset()
            else:
                _LOGGER.warning("audio-stop received without any audio-chunk")

            if self.stream_transcript:
                await self.write_event(TranscriptStop().event())

            await self.write_event(
                Transcript(text=text, language=self.language).event()
            )
            _LOGGER.debug("Sent final transcript (%d chars)", len(text))
        finally:
            if self._model_lock is not None and self._model_lock.locked():
                self._model_lock.release()
                _LOGGER.debug("Model lock released")
            self._model_lock = None
            self._locked_model_name = None

    async def disconnect(self) -> None:
        """Reset model state and buffer when the client goes away.

        Called unconditionally by AsyncEventHandler.run() — also when the
        client drops mid-stream without sending audio-stop. The model lock
        is released here as a safety net (finally-sicher, AUDIT-003).
        """
        self.audio_buffer = np.array([], dtype=np.float32)
        if self.speech2text is not None:
            try:
                self.speech2text.reset()
            except Exception:
                _LOGGER.exception("model.reset() failed during disconnect")
              
        if self._model_lock is not None and self._model_lock.locked():
            try:
                self._model_lock.release()
                _LOGGER.debug("Model lock released in disconnect")
            except RuntimeError:
                pass
        self._model_lock = None
        self._locked_model_name = None
        _LOGGER.debug("Client disconnected, state reset")
