"""
Recording Trigger — Raspberry Pi 3B+ Pipeline

Listens for on-demand "record" commands from the Android application,
either via the WebSocket connection or a dedicated HTTP POST endpoint.

When the app sends a record signal the pipeline:
  1. Records MIC audio for MAX_RECORDING_SECONDS (or until the app sends stop).
  2. Runs the full preprocessing → feature-extraction → prediction pipeline.
  3. Pushes the result back through the WebSocket and saves to DB.

The Android developer can customise behaviour via config.py:
  - MAX_RECORDING_SECONDS
  - APP_API_PORT (WS)   / APP_API_PORT+1 (HTTP)
"""

import asyncio
import json
import wave
import logging
import struct
from datetime import datetime
from pathlib import Path

from config import (
    SAMPLE_RATE,
    CHANNELS,
    BIT_DEPTH,
    MAX_RECORDING_SECONDS,
    RECORDINGS_DIR,
    LISTEN_CHUNK,
    APP_API_HOST,
    APP_API_PORT,
)

log = logging.getLogger(__name__)


class RecordingTrigger:
    """Accept record-start / record-stop commands from the Android app."""

    def __init__(self, pipeline_callback):
        """
        Parameters
        ----------
        pipeline_callback : async callable(wav_path: str) → dict
            The pipeline's ``process_file`` coroutine.
        """
        self._callback = pipeline_callback
        self._recording = False
        self._pyaudio = None

    # ── HTTP handler (POST /record) ──────────────────────────────────────

    async def handle_http_record(self, request):
        """
        POST /record  { "action": "start" | "stop", "duration": 10 }
        """
        from aiohttp import web

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        action = body.get("action", "start")
        duration = min(int(body.get("duration", MAX_RECORDING_SECONDS)), MAX_RECORDING_SECONDS)

        if action == "start":
            if self._recording:
                return web.json_response({"status": "already_recording"})
            result = await self._do_record(duration)
            return web.json_response(result)

        return web.json_response({"status": "ok"})

    # ── WebSocket message handler ────────────────────────────────────────

    async def handle_ws_message(self, message: str):
        """
        Incoming WS message: { "command": "record", "duration": 10 }
        Returns prediction dict or None.
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return None

        if data.get("command") == "record":
            duration = min(int(data.get("duration", MAX_RECORDING_SECONDS)), MAX_RECORDING_SECONDS)
            return await self._do_record(duration)
        return None

    # ── actual recording ─────────────────────────────────────────────────

    async def _do_record(self, duration: int) -> dict:
        self._recording = True
        wav_path = str(
            RECORDINGS_DIR
            / f"app_record_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.wav"
        )

        log.info("App-triggered recording  [%d s → %s]", duration, wav_path)

        # Run blocking PyAudio capture in executor to keep event loop responsive
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._capture_wav, wav_path, duration)
            result = await self._callback(wav_path)
        except Exception as exc:
            log.error("Recording failed: %s", exc)
            result = {"error": str(exc)}
        finally:
            self._recording = False

        return result

    def _capture_wav(self, path: str, duration: int):
        """Blocking microphone capture → WAV file."""
        import pyaudio

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=LISTEN_CHUNK,
            )
            frames = []
            total_chunks = int(SAMPLE_RATE / LISTEN_CHUNK * duration)
            for _ in range(total_chunks):
                frames.append(stream.read(LISTEN_CHUNK, exception_on_overflow=False))
            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()

        with wave.open(path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(BIT_DEPTH // 8)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))

        log.info("WAV written  [%s]", path)
