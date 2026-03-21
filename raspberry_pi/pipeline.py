"""
Main Pipeline — Raspberry Pi 3B+ CryingSense

Orchestrates the full loop:
  1. Listen on microphone → .wav
  2. Audio preprocessing (normalise, trim, resample)
  3. Feature extraction (4-channel mel-spectrogram tensor)
  4. Quantized CNN inference
  5. Push result to:
        • Android app  (WebSocket)
        • MongoDB       (audio_files, audio_sessions, cry_classifications, device_registrations)

Runs an asyncio event loop with:
  - continuous MIC listener (blocking capture in thread-pool)
  - WebSocket + HTTP server for Android app communication
  - on-demand recording trigger from app

Designed for RPi 3B+ constraints:
  - 1 GB RAM  → small pool sizes, no large in-memory buffers
  - 4 × Cortex-A53 @ 1.4 GHz → ONNX runtime pinned to 4 threads
  - 32 GB SD  → recordings auto-cleaned after DB upload
"""

import os
import sys
import wave
import uuid
import signal
import asyncio
import logging
from pathlib import Path
from datetime import datetime

from config import (
    MODEL_PATH,
    NUM_CLASSES,
    SAMPLE_RATE,
    CHANNELS,
    BIT_DEPTH,
    DURATION,
    PREDICTION_INTERVAL,
    LISTEN_CHUNK,
    MIC_DEVICE_INDEX,
    RECORDINGS_DIR,
    DEVICE_ID,
    LOG_LEVEL,
    APP_API_HOST,
    APP_API_PORT,
)

# Pipeline components
from audio_preprocessor import AudioPreprocessor
from feature_extractor import FeatureExtractor
from model_loader import ModelLoader
from predictor import Predictor
from database_handler import DatabaseHandler
from app_notifier import AppNotifier
from recording_trigger import RecordingTrigger

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_device_id() -> str:
    """Return configured DEVICE_ID or derive one from MAC address."""
    if DEVICE_ID:
        return DEVICE_ID
    try:
        mac = uuid.getnode()
        return f"rpi-{mac:012x}"
    except Exception:
        return f"rpi-{uuid.uuid4().hex[:12]}"


def _capture_chunk(pa_instance, duration: float, path: str):
    """Blocking: record *duration* seconds from the default mic to *path*."""
    import pyaudio

    stream = pa_instance.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=MIC_DEVICE_INDEX,
        frames_per_buffer=LISTEN_CHUNK,
    )
    frames = []
    total = int(SAMPLE_RATE / LISTEN_CHUNK * duration)
    for _ in range(total):
        frames.append(stream.read(LISTEN_CHUNK, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()

    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BIT_DEPTH // 8)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))


# ── Pipeline ─────────────────────────────────────────────────────────────────

class CryingSensePipeline:
    """Top-level orchestrator."""

    def __init__(self):
        self.device_id = _get_device_id()
        self.session_id: str = ""

        # Components
        self.preprocessor = AudioPreprocessor()
        self.extractor = FeatureExtractor()
        self.model = ModelLoader(MODEL_PATH, NUM_CLASSES).load()
        self.predictor = Predictor(self.model)
        self.db = DatabaseHandler()
        self.notifier: AppNotifier = None  # type: ignore[assignment]
        self.recorder: RecordingTrigger = None  # type: ignore[assignment]

        self._running = False

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        log.info("═" * 60)
        log.info("CryingSense Pipeline — Raspberry Pi 3B+")
        log.info("Device : %s", self.device_id)
        log.info("Model  : %s  [%s]", MODEL_PATH, self.model.backend)
        log.info("═" * 60)

        # DB
        self.db.connect()
        self.db.register_device(self.device_id)
        self.session_id = self.db.start_session(self.device_id)

        # App communication
        self.notifier = AppNotifier(self.device_id, self.session_id)
        self.recorder = RecordingTrigger(self.process_file)
        self.notifier.set_recording_trigger(self.recorder)
        await self.notifier.start()

        # Start HTTP /record endpoint alongside existing HTTP server
        await self._start_record_endpoint()

        self._running = True

        # Graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.ensure_future(self.stop()))
            except NotImplementedError:
                pass  # Windows

        # Main loop
        await self._listen_loop()

    async def stop(self):
        if not self._running:
            return
        self._running = False
        log.info("Shutting down…")
        self.db.end_session(self.session_id)
        self.db.close()
        await self.notifier.stop()

    # ── HTTP /record endpoint ────────────────────────────────────────────

    async def _start_record_endpoint(self):
        try:
            from aiohttp import web

            app = web.Application()
            app.router.add_post("/record", self.recorder.handle_http_record)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, APP_API_HOST, APP_API_PORT + 2)
            await site.start()
            log.info("POST /record  on port %d", APP_API_PORT + 2)
        except ImportError:
            log.warning("aiohttp not installed — POST /record disabled")

    # ── main loop ────────────────────────────────────────────────────────

    async def _listen_loop(self):
        """Continuously capture → preprocess → predict → upload."""
        import pyaudio

        pa = pyaudio.PyAudio()
        loop = asyncio.get_running_loop()

        log.info("Listening…  (%.1f s chunks, every %.1f s)", DURATION, PREDICTION_INTERVAL)

        try:
            while self._running:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                wav_path = str(RECORDINGS_DIR / f"chunk_{ts}.wav")

                # Blocking capture in thread-pool
                await loop.run_in_executor(
                    None, _capture_chunk, pa, DURATION, wav_path
                )

                # Process
                result = await self.process_file(wav_path)

                # Only persist actual cries
                if result.get("is_cry"):
                    await self._persist(wav_path, result)
                    await self.notifier.notify(result)
                else:
                    # Clean up noise chunks immediately
                    self._remove_file(wav_path)

                # Heartbeat
                self.db.heartbeat(self.device_id)

                # Gap between captures
                gap = PREDICTION_INTERVAL - DURATION
                if gap > 0:
                    await asyncio.sleep(gap)

        except asyncio.CancelledError:
            pass
        finally:
            pa.terminate()

    # ── single-file processing (also used by RecordingTrigger) ───────────

    async def process_file(self, wav_path: str) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._process_sync, wav_path)

    def _process_sync(self, wav_path: str) -> dict:
        audio = self.preprocessor.preprocess_file(wav_path)
        features = self.extractor.extract(audio)
        return self.predictor.predict(features, audio_file=wav_path)

    # ── DB persistence ───────────────────────────────────────────────────

    async def _persist(self, wav_path: str, result: dict):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._persist_sync, wav_path, result)

    def _persist_sync(self, wav_path: str, result: dict):
        # 1. classification
        cid = self.db.save_classification(
            result=result,
            device_id=self.device_id,
            session_id=self.session_id,
        )

        # 2. audio file
        if cid:
            self.db.save_audio_file(
                wav_path=wav_path,
                device_id=self.device_id,
                session_id=self.session_id,
                duration_seconds=DURATION,
                classification_id=cid,
            )

    # ── cleanup ──────────────────────────────────────────────────────────

    @staticmethod
    def _remove_file(path: str):
        try:
            os.remove(path)
        except OSError:
            pass


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    pipeline = CryingSensePipeline()
    try:
        asyncio.run(pipeline.start())
    except KeyboardInterrupt:
        log.info("Interrupted")


if __name__ == "__main__":
    main()
