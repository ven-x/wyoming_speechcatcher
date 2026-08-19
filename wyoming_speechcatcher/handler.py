"""Wyoming event handler for Speechcatcher ASR.

Implements the per-connection event flow:

- ``describe``     → reply with ``info`` (AsrProgram advertisement)
- ``transcribe``   → remember requested language/model name
- ``audio-start``  → reset per-utterance state (buffer, model state) and
  acquire the per-model lock. A repeated ``audio-start`` on a connection
  that already holds the lock is treated as a plain reset without
  re-acquiring the lock (AUDIT-024 — ``asyncio.Lock`` is not reentrant,
  a second acquire would deadlock this handler and DoS the model).
  Lock acquisition is bounded by ``LOCK_ACQUIRE_TIMEOUT`` (30 s,
  AUDIT-024-2): on timeout the handler sets ``_model_load_failed`` and
  answers ``audio-stop`` with an empty transcript instead of blocking
  forever.
- ``audio-chunk``  → convert to 16 kHz mono int16, buffer until at least
  ``MIN_BUFFER_SAMPLES`` samples are available, then run streaming
  recognition. ``None`` results (chunk too short after frame trimming)
  are ignored.
- ``audio-stop``   → pad a short remainder with silence up to
  ``FINAL_MIN_SAMPLES`` (a too-short final chunk would crash the
  encoder's Conv2d subsampling), finalize with ``is_final=True``, send
  ``transcript`` (plus ``transcript-stop`` when streaming), reset the
  model, close the connection (return ``False``).
- ``disconnect()`` → always reset the model and clear the buffer, so a
  client that drops mid-stream never leaves dirty decoder state behind.

Optional partial transcripts (``transcript-chunk``) are emitted when the
server was started with ``--stream-transcript``.

The module is importable without the heavy ``speechcatcher``/``torch``
dependencies — only numpy and wyoming are required at import time. The
model object itself is supplied by :class:`wyoming_speechcatcher.state.State`
via dependency injection, which keeps the handler unit-testable with
mocks.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

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

# Python 3.8 compatibility: asyncio.to_thread was added in 3.9.
# pyproject.toml declares requires-python >=3.8, so we provide a shim.
if sys.version_info >= (3, 9):
    _to_thread = asyncio.to_thread  # type: ignore[attr-defined]
else:

    async def _to_thread(func, *args, **kwargs):  # type: ignore[no-redef]
        """Backport of asyncio.to_thread for Python 3.8."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

# Sample rate / format the model expects (PCM int16, mono, 16 kHz).
MODEL_RATE = 16000
MODEL_WIDTH = 2
MODEL_CHANNELS = 1

# Minimum number of samples before a chunk is fed to the model in
# streaming mode (is_final=False).
#
# Why 8000 and not win_length (400): the STFT frontend (win_length=400,
# hop_length=160) produces frames that are trimmed at chunk boundaries
# (2 frames per side for middle chunks = 4 frames total trim). After
# trimming, the encoder's Conv2dSubsampling (factor 4) needs enough
# frames to produce at least a few encoder time steps.
#
# At N=1000 (the previous value): 4 frames → 0 frames after trim →
# the chunk is skipped entirely (apply_frontend returns None). This
# means most streaming chunks were discarded, and only the final chunk
# (is_final=True) with the accumulated buffer produced any output —
# but even that buffer was too short for good recognition.
#
# At N=8000 (0.5 s @ 16 kHz): 48 frames → 44 frames after trim → 11
# frames after subsampling — plenty for the decoder to work with.
# This matches the order of magnitude of the upstream speechcatcher
# server (8192 samples per chunk, speechcatcher_server.py).
MIN_BUFFER_SAMPLES = 8000

# Minimum length of the final buffer passed to the model with
# is_final=True on audio-stop. Shorter remainders are padded with
# digital silence up to this length. Must be large enough that the
# STFT frontend produces enough frames for the encoder's Conv2dSubsampling
# after trimming (see MIN_BUFFER_SAMPLES rationale).
FINAL_MIN_SAMPLES = 8000

# Maximum buffer size as a safety valve against unbounded memory growth
# when a client streams without ever sending audio-stop. 30 s @ 16 kHz.
MAX_BUFFER_SAMPLES = 30 * MODEL_RATE

# Maximum time (seconds) to wait for the per-model lock before giving up
# (AUDIT-024-2). Rationale for 30 s: the lock is held for one complete
# utterance (audio-start → audio-stop). A legitimate utterance is bounded
# in practice — the audio buffer itself is capped at 30 s of audio
# (MAX_BUFFER_SAMPLES), and Home Assistant's own STT timeouts are in the
# same order of magnitude. A lock holder blocking longer than 30 s is
# therefore pathological (e.g. a stuck/slow client without auth —
# Wyoming has none, so any network peer can trigger this). Without a
# bound, such a client would block every other connection on the same
# model until the server restarts (DoS). On timeout we degrade to the
# AUDIT-006 "controlled completion" path instead: the connection stays
# usable and audio-stop returns an empty Transcript(text="").
LOCK_ACQUIRE_TIMEOUT = 30.0


def build_info(args: Any, state: "State") -> Info:
    """Build the Wyoming ``info`` event advertising this ASR service.

    Args:
        args: Parsed argparse namespace. Expected attributes:
            ``stream_transcript`` (bool), ``external_vad`` (bool).
        state: Shared :class:`~wyoming_speechcatcher.state.State`. Used to
            report truthful per-model ``installed`` status (AUDIT-005).

    Returns:
        An :class:`wyoming.info.Info` describing the speechcatcher
        program and all known models (see STATUS.md "Wyoming Info-Objekt").
    """
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
    """Handle one Wyoming client connection.

    Args:
        args: Parsed argparse namespace (uses ``stream_transcript``).
        state: Shared :class:`~wyoming_speechcatcher.state.State` used for
            lazy model resolution via ``state.get_model()``.
        reader: Asyncio stream reader for the client connection.
        writer: Asyncio stream writer for the client connection.
    """

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

        # Info event is identical for every describe on this connection.
        self._info_event = build_info(args, state).event()

        # Normalizes incoming audio to 16 kHz / int16 / mono.
        self.converter = AudioChunkConverter(
            rate=MODEL_RATE, width=MODEL_WIDTH, channels=MODEL_CHANNELS
        )

        # Per-request selections (set by the transcribe event).
        self.language: Optional[str] = None
        self.model_name: Optional[str] = None

        # Lazily acquired model (shared object owned by State).
        self.speech2text: Optional[Any] = None

        # AUDIT-006: Flag, das anzeigt, dass die Modell-Akquise in
        # _handle_audio_chunk fehlgeschlagen ist. _handle_audio_stop
        # sendet dann ein leeres Transcript(text="") statt unkontrolliert
        # abzubrechen.
        self._model_load_failed: bool = False

        # AUDIT-003 (Variante B): Lock für exklusiven Modell-Zugriff.
        # Wird bei audio-start akquiriert, bei audio-stop/disconnect freigegeben.
        self._model_lock: Optional[asyncio.Lock] = None

        # AUDIT-027-2: Der Modell-Kurzname, für den _model_lock aktuell
        # gehalten wird. Wird bei erfolgreicher Akquise (audio-start oder
        # Fallback in _handle_audio_chunk) gesetzt und beim Release
        # (audio-stop finally / disconnect / Transcribe-Modellwechsel)
        # zurückgesetzt auf None. Der Transcribe-Zweig vergleicht den
        # neu aufgelösten Modellnamen gegen dieses Attribut, um einen
        # Modellwechsel unter gehaltenem Lock zu erkennen (AUDIT-027-1).
        self._locked_model_name: Optional[str] = None

        # Float32 sample buffer, normalized to [-1.0, 1.0].
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
            # AUDIT-006: transcribe.name sofort gegen bekannte Short-Tags
            # validieren. Ungueltige Namen werden verworfen (Fallback auf
            # Sprach-Default), statt spaeter in get_model() einen
            # ValueError auszuloesen.
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

            # AUDIT-027-1 (Variante a): Wenn diese Verbindung bereits ein
            # Modell-Lock hält und das neue transcribe ein *anderes*
            # Modell auflöst als das gelockte, muss das alte Lock
            # freigegeben und das Lock für das neue Modell akquiriert
            # werden. Sonst würde die nachfolgende Inferenz (Fallback in
            # _handle_audio_chunk prüft nur `self._model_lock is None`,
            # nicht ob das Lock zum aufgelösten Modell passt) auf dem
            # neuen Modell laufen, OHNE dessen Lock zu halten — eine
            # parallele Verbindung auf dem neuen Modell könnte
            # gleichzeitig inferieren → Decoder-State-Vermischung (genau
            # das, was AUDIT-003 verhindern soll). Der Angriffsvektor
            # existiert, weil Wyoming keine Auth hat und die Sequenz
            # transcribe(A)→audio-start→transcribe(B)→audio-chunk
            # protokollgemäß möglich ist (HA sendet transcribe vor
            # audio-start; Mehrsprachen-Haushalte teilen sich einen
            # Server).
            #
            # Variante a (gewählt) ist protokoll-kompatibel: altes Lock
            # freigeben, speech2text zurücksetzen (Decoder-State des
            # alten Modells verwerfen), Buffer leeren, neues Lock
            # akquirieren. HA schickt ein ggf. erneutes transcribe, wenn
            # es die Pipeline neu startet — wir bleiben damit in der
            # erwarteten State-Machine. Variante b (transcribe
            # verwerfen, altes Modell beibehalten) wäre der kleinere
            # Eingriff gewesen, würde aber das transcribe-Event
            # stillschweigend ignorieren und somit die HA-Pipeline
            # unerwartet bremsen.
            #
            # resolve_model_name kann ValueError werfen (unbekannter
            # Tag nach Sprach-Default-Auflösung) — in dem Fall können wir
            # keinen sauberen Modellwechsel durchführen. Wir loggen die
            # Warning und lassen das bisherige Lock unverändert bestehen
            # (Variante b als Fallback für diesen fehlerhaften Fall);
            # _handle_audio_chunk wird später den Ladeversuch
            # kontrolliert abbrechen (AUDIT-006).
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
                    # Altes Lock freigeben. reset() des alten Modells
                    # ist nicht nötig: der Decoder-State des alten
                    # Modells ist durch das Lock geschützt — ein anderer
                    # Client auf dem alten Modell wird uns nicht
                    # stören, und wir selbst geben das Modell hier auf.
                    # Das Modell-Objekt verbleibt im State-Cache.
                    self._model_lock.release()
                    self._model_lock = None
                    old_locked = self._locked_model_name
                    self._locked_model_name = None

                    # speech2text auf None setzen, damit
                    # _handle_audio_chunk das neue Modell lädt. Buffer
                    # leeren (altes Audio gehört zum alten Modell /
                    # Decoder-State).
                    self.speech2text = None
                    self.audio_buffer = np.array([], dtype=np.float32)

                    # Neues Lock akquirieren (mit Zeitlimit,
                    # AUDIT-024-2). Schlägt die Akquise fehl (Timeout),
                    # degradiert der Handler wie gewohnt in den
                    # _model_load_failed-Pfad.
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
            # AUDIT-024 (Variante a): Erneutes audio-start auf einer
            # Verbindung, die das Modell-Lock bereits haelt, ist der
            # "Client beginnt neu"-Fall (protokollwidrig, aber ohne Auth
            # vom Netz ausloesbar). asyncio.Lock ist NICHT reentrant —
            # ein zweites acquire() auf dasselbe Lock wuerde diesen
            # Handler dauerhaft blockieren: wyomings run() awaited
            # handle_event seriell, d.h. die Verbindung liest nie wieder
            # vom Socket, disconnect() (das Safety-Net) wird nie erreicht
            # und das Lock bleibt bis zum Server-Neustart gehalten
            # (DoS fuer alle Verbindungen auf dasselbe Modell).
            # Deshalb: bereits gehaltenes Lock NICHT erneut akquirieren,
            # sondern das wiederholte audio-start als Reset behandeln —
            # Buffer leeren + Decoder-State zuruecksetzen, Lock bleibt
            # exakt einmal gehalten.
            if self._model_lock is not None and self._model_lock.locked():
                _LOGGER.warning(
                    "Repeated audio-start on a connection that already holds "
                    "the model lock — treating as reset (no re-acquire)"
                )
                self.audio_buffer = np.array([], dtype=np.float32)
                if self.speech2text is not None:
                    self.speech2text.reset()
                return True

            # Fresh utterance: drop any leftovers from a previous stream.
            self.audio_buffer = np.array([], dtype=np.float32)
            if self.speech2text is not None:
                self.speech2text.reset()

            # AUDIT-003 (Variante B): Lock für exklusiven Modell-Zugriff
            # akquirieren. Die komplette Äußerung (audio-start → audio-stop)
            # hält das Modell exklusiv, damit parallele Verbindungen den
            # Decoder-State nicht vermischen.
            # AUDIT-024-2: Akquise mit Zeitlimit — bei Timeout wird
            # _model_load_failed gesetzt (kontrollierter Abschluss mit
            # leerem Transcript bei audio-stop statt ewigem Blockieren).
            resolved = self.state.resolve_model_name(self.language, self.model_name)
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
        """Acquire the per-model lock with a bounded wait (AUDIT-024-2).

        Wraps ``lock.acquire()`` in ``asyncio.wait_for`` with
        :data:`LOCK_ACQUIRE_TIMEOUT` (30 s, rationale at the constant).
        On success ``self._model_lock`` holds the acquired lock.

        On timeout the handler degrades to the AUDIT-006 controlled-
        completion path instead of blocking forever: ``_model_load_failed``
        is set, ``_model_lock`` stays ``None`` (nothing was acquired, so
        nothing must be released), ``_handle_audio_chunk`` drops further
        audio and ``_handle_audio_stop`` answers with an empty
        ``Transcript(text="")``. This keeps a stuck lock holder from
        wedging every other connection on the same model until restart
        (DoS via the unauthenticated Wyoming protocol).

        Args:
            resolved: Resolved model short tag (from
                ``state.resolve_model_name``) whose lock should be taken.
        """
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
            # AUDIT-027-2: Tracking zurücksetzen (defense in depth;
            # Aufrufer sollte _locked_model_name bereits geleert haben).
            self._locked_model_name = None
            return
        self._model_lock = lock
        # AUDIT-027-2: Das erfolgreich gelockte Modell merken, damit der
        # Transcribe-Zweig einen Modellwechsel unter gehaltenem Lock
        # erkennt (AUDIT-027-1).
        self._locked_model_name = resolved
        _LOGGER.debug("Model lock acquired for %s", resolved)

    async def _handle_audio_chunk(self, chunk: AudioChunk) -> None:
        """Buffer a chunk and run streaming recognition when enough samples.

        Both the lazy model acquisition (``state.get_model``) and the
        inference call (``self.speech2text(...)``) run in a worker thread
        via ``_to_thread`` so they never block the asyncio event loop
        (AUDIT-002).

        Downmix-Strategie (AUDIT-010): genau eine, und das ist der
        ``AudioChunkConverter`` (``channels=1`` erzwungen via
        ``audioop.tomono`` — Summe L+R, geclippt). Nach
        ``self.converter.convert(chunk)`` ist der Chunk garantiert mono;
        es gibt bewusst KEINEN manuellen ``audio[::channels]``-Pfad mehr
        im Handler (frueherer doppelter Downmix entfernt). Ein manueller
        Downmix waere zudem semantisch falsch: er wuerde — anders als der
        Converter — auf bereits resampletes 16-kHz-Audio angewendet und
        Samples verwerfen statt Kanaele zu mischen.
        """
        # AUDIT-006: Nach einem fehlgeschlagenen Ladeversuch keine
        # erneuten Versuche pro Chunk (verhindert wiederholte Downloads/
        # RAM-Allokationen bei einem Stream, der ohnehin scheitert).
        # Der Client bekommt bei audio-stop ein leeres Transcript.
        if self._model_load_failed:
            return

        if self.speech2text is None:
            # Model loading can take seconds (RAM load or download from
            # HuggingFace). Run it off the event loop.
            # AUDIT-006: Modellauflösungs- und Ladefehler (ValueError bei
            # ungültigem/nicht erlaubtem Namen, RuntimeError bei Lade-/
            # Netzwerkfehlern) kontrolliert behandeln: Fehler loggen,
            # Ladefehler-Flag setzen — _handle_audio_stop sendet dann ein
            # leeres Transcript(text="") statt die Verbindung unkontrolliert
            # abbrechen zu lassen.
            try:
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

            # AUDIT-003 (Variante B): Falls audio-start noch nicht kam
            # (Fallback), Lock hier akquirieren. Normalerweise wurde das
            # Lock bereits bei audio-start gesetzt.
            # AUDIT-024-2: ebenfalls mit Zeitlimit (siehe
            # _acquire_model_lock) — bei Timeout ist _model_load_failed
            # gesetzt und dieser Chunk (und alle weiteren) werden
            # verworfen; audio-stop liefert das leere Transcript.
            if self._model_lock is None and not self._model_load_failed:
                resolved = self.state.resolve_model_name(
                    self.language, self.model_name
                )
                await self._acquire_model_lock(resolved)
                if self._model_load_failed:
                    # Lock-Akquise ist in den Timeout gelaufen — kein
                    # exklusiver Modellzugriff, also auch keine Inferenz.
                    return

        chunk = self.converter.convert(chunk)

        # int16 bytes → float32 in [-1.0, 1.0] (matches speechcatcher_server).
        # AUDIT-010: chunk ist hier garantiert mono — der Downmix ist
        # ausschliesslich Aufgabe des AudioChunkConverter (siehe Docstring).
        audio_i16 = np.frombuffer(chunk.audio, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32767.0

        self.audio_buffer = np.concatenate([self.audio_buffer, audio_f32])
        if len(self.audio_buffer) > MAX_BUFFER_SAMPLES:
            # Safety valve: keep the tail so a never-ending stream cannot
            # exhaust memory (client should have sent audio-stop by then).
            _LOGGER.warning(
                "Audio buffer exceeded %d samples, dropping oldest audio",
                MAX_BUFFER_SAMPLES,
            )
            self.audio_buffer = self.audio_buffer[-MAX_BUFFER_SAMPLES:]

        if len(self.audio_buffer) < MIN_BUFFER_SAMPLES:
            # Not enough samples for a frontend frame — keep buffering.
            return

        speech = self.audio_buffer
        self.audio_buffer = np.array([], dtype=np.float32)

        # Inference runs synchronously inside Speech2TextStreaming; keep
        # the event loop free for other connections (AUDIT-002).
        result = await _to_thread(self.speech2text, speech=speech, is_final=False)
        if not result:
            # Chunk produced no decodable frame yet — nothing to stream.
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
                # AUDIT-006 / AUDIT-024-2: Modell-Akquise oder
                # Lock-Akquise ist fehlgeschlagen (geloggt). Kontrollierter
                # Abschluss: leeres Transcript statt Verbindungsabbruch —
                # HA zeigt dann "nichts erkannt" statt Timeout.
                # WICHTIG: Dieser Zweig muss VOR `speech2text is not None`
                # geprüft werden: Im Fallback-Pfad (_handle_audio_chunk
                # ohne vorheriges audio-start) wird das Modell ZUERST
                # geladen und DANACH das Lock akquiriert — läuft die
                # Lock-Akquise in den AUDIT-024-2-Timeout, ist speech2text
                # bereits gesetzt, aber OHNE exklusiven Zugriff. Eine
                # finale Inferenz würde dann den Decoder-State des
                # Lock-Inhabers vermischen (genau das, was AUDIT-003
                # verhindern soll). Deshalb: kein Modell-Call, kein
                # reset() — nur das leere Transcript.
                _LOGGER.warning(
                    "audio-stop after failed model/lock acquisition "
                    "(language=%s, model=%s) — sending empty transcript",
                    self.language,
                    self.model_name,
                )
            elif self.speech2text is not None:
                speech = self.audio_buffer
                self.audio_buffer = np.array([], dtype=np.float32)

                # A too-short final buffer (0..FINAL_MIN_SAMPLES-1 samples,
                # e.g. a one-word utterance) would produce fewer STFT frames
                # than the encoder's Conv2d subsampling needs (3x3 kernel,
                # stride 2, twice -> >= 4 frames) and crash with
                # "RuntimeError: Kernel size can't be greater than actual
                # input size". Pad with digital silence up to a safe length
                # before the is_final=True call — the same thing
                # speechcatchers own server does on EOF
                # (np.zeros(1000, dtype=np.int16), speechcatcher_server.py).
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
                    # Nothing left in the buffer (all audio was already
                    # processed via audio-chunk, or no audio arrived at all).
                    # Send pure silence so the model can finalize its decoder
                    # state instead of crashing on an empty/1-frame input.
                    _LOGGER.debug(
                        "audio-stop with empty buffer — sending %d samples of silence",
                        FINAL_MIN_SAMPLES,
                    )
                    speech = np.zeros(FINAL_MIN_SAMPLES, dtype=np.float32)

                # finalize_all wird nur vom native-Decoder unterstützt
                # (speechcatcher.Speech2TextStreaming). Der ESPnet-Decoder
                # (espnet_streaming_decoder) kennt dieses Keyword-Argument
                # nicht und würde mit TypeError abstürzen.
                call_kwargs: dict = {"speech": speech, "is_final": True}
                if getattr(self.args, "decoder", "native") == "native":
                    call_kwargs["finalize_all"] = True
                result = await _to_thread(self.speech2text, **call_kwargs)
                if result:
                    # First n-best hypothesis, text element.
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
            # AUDIT-003 (Variante B): Lock garantiert freigeben.
            if self._model_lock is not None and self._model_lock.locked():
                self._model_lock.release()
                _LOGGER.debug("Model lock released")
            self._model_lock = None
            # AUDIT-027-2: Locked-Modell-Tracking zurücksetzen.
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
            except Exception:  # pragma: no cover - defensive
                _LOGGER.exception("model.reset() failed during disconnect")
        # AUDIT-003 (Variante B): Lock garantiert freigeben.
        if self._model_lock is not None and self._model_lock.locked():
            try:
                self._model_lock.release()
                _LOGGER.debug("Model lock released in disconnect")
            except RuntimeError:  # pragma: no cover - lock already released
                pass
        self._model_lock = None
        # AUDIT-027-2: Locked-Modell-Tracking zurücksetzen.
        self._locked_model_name = None
        _LOGGER.debug("Client disconnected, state reset")
