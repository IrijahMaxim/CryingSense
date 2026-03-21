"""
App Notifier — Raspberry Pi 3B+ Pipeline

Sends prediction results to the Android application via:
  - WebSocket (real-time push on port APP_API_PORT)
  - Lightweight HTTP endpoint for health-check / status

The Android developer can customise APP_API_HOST and APP_API_PORT
in config.py or via environment variables.

Data pushed per cry event
-------------------------
{
  "type": "prediction",
  "prediction": "hunger",
  "confidence": 0.92,
  "timestamp": "2026-03-21T12:00:00Z",
  "time_end": "2026-03-21T12:00:05Z",
  "device_id": "rpi-001",
  "session_id": "...",
  "alert": true
}
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Set

from config import APP_API_HOST, APP_API_PORT, DEVICE_TYPE, DURATION

log = logging.getLogger(__name__)

# Connected WebSocket clients
_clients: Set = set()

# Lazily loaded modules (avoid import errors when websockets isn't installed yet)
_ws_module = None
_aiohttp_web = None


def _load_websockets():
    global _ws_module
    if _ws_module is None:
        import websockets  # type: ignore
        _ws_module = websockets
    return _ws_module


class AppNotifier:
    """Push predictions to connected Android clients."""

    def __init__(self, device_id: str, session_id: str = ""):
        self.device_id = device_id
        self.session_id = session_id
        self._server = None
        self._http_runner = None
        self._recording_trigger = None  # set via set_recording_trigger()

    def set_recording_trigger(self, trigger):
        """Wire in the RecordingTrigger so WS messages can be routed."""
        self._recording_trigger = trigger

    # ── WebSocket handler ────────────────────────────────────────────────

    async def _ws_handler(self, ws):
        _clients.add(ws)
        remote = ws.remote_address
        log.info("WS client connected  [%s]", remote)
        try:
            async for msg in ws:
                if self._recording_trigger is not None:
                    result = await self._recording_trigger.handle_ws_message(msg)
                    if result is not None:
                        await self._safe_send(ws, json.dumps(result))
        finally:
            _clients.discard(ws)
            log.info("WS client disconnected  [%s]", remote)

    # ── HTTP health-check ────────────────────────────────────────────────

    async def _http_app(self):
        from aiohttp import web

        async def _status(request):
            return web.json_response(
                {
                    "status": "ok",
                    "device_id": self.device_id,
                    "device_type": DEVICE_TYPE,
                    "session_id": self.session_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )

        app = web.Application()
        app.router.add_get("/status", _status)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, APP_API_HOST, APP_API_PORT + 1)
        await site.start()
        self._http_runner = runner
        log.info("HTTP /status listening on %s:%d", APP_API_HOST, APP_API_PORT + 1)

    # ── start / stop ─────────────────────────────────────────────────────

    async def start(self):
        ws_mod = _load_websockets()
        self._server = await ws_mod.serve(
            self._ws_handler, APP_API_HOST, APP_API_PORT
        )
        log.info("WebSocket server listening on ws://%s:%d", APP_API_HOST, APP_API_PORT)

        try:
            await self._http_app()
        except ImportError:
            log.warning("aiohttp not installed — HTTP /status endpoint disabled")

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._http_runner:
            await self._http_runner.cleanup()

    # ── push prediction ──────────────────────────────────────────────────

    async def notify(self, result: dict):
        """Broadcast a prediction result to all connected clients."""
        if not _clients:
            return

        ts = result.get("timestamp", datetime.utcnow().isoformat() + "Z")
        # Compute time_end = timestamp + DURATION
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            time_end = (dt + timedelta(seconds=DURATION)).isoformat().replace("+00:00", "Z")
        except Exception:
            time_end = ts

        payload = json.dumps(
            {
                "type": "prediction",
                "prediction": result.get("prediction", "unknown"),
                "confidence": result.get("confidence", 0),
                "timestamp": ts,
                "time_end": time_end,
                "device_id": self.device_id,
                "session_id": self.session_id,
                "alert": result.get("alert", False),
            }
        )

        ws_mod = _load_websockets()
        await asyncio.gather(
            *[self._safe_send(ws, payload) for ws in list(_clients)],
            return_exceptions=True,
        )

    @staticmethod
    async def _safe_send(ws, payload: str):
        try:
            await ws.send(payload)
        except Exception:
            _clients.discard(ws)
