#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "wyoming==1.6.0",
#   "mlx-whisper==0.4.1",
#   "mlx-audio",
#   "numpy",
#   "pip",
# ]
# ///

import asyncio
import traceback
from dataclasses import dataclass

import numpy as np
import mlx_whisper
from mlx_audio.tts.utils import load_model as load_kokoro

from wyoming.event import Event
from wyoming.server import AsyncServer, AsyncEventHandler

from wyoming.asr import Transcript
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.tts import Synthesize


# Hugging Face model repo IDs (must exist)
WHISPER_MODEL_ID = "mlx-community/whisper-turbo"
KOKORO_MODEL_ID = "mlx-community/Kokoro-82M-bf16"

# Network
STT_BIND = "tcp://0.0.0.0:10900"
TTS_BIND = "tcp://0.0.0.0:10800"

# Audio assumptions for Home Assistant Wyoming STT:
# HA typically sends 16kHz mono 16-bit PCM for STT.
DEFAULT_STT_RATE = 16000
DEFAULT_STT_WIDTH = 2
DEFAULT_STT_CHANNELS = 1

# Kokoro output PCM
TTS_RATE = 24000
TTS_WIDTH = 2
TTS_CHANNELS = 1


@dataclass
class AudioFormat:
    rate: int = DEFAULT_STT_RATE
    width: int = DEFAULT_STT_WIDTH
    channels: int = DEFAULT_STT_CHANNELS


KOKORO_MODEL = None  # loaded at startup


def _pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """Convert little-endian signed PCM16 bytes to float32 in [-1, 1)."""
    if not audio_bytes:
        return np.zeros((0,), dtype=np.float32)
    pcm = np.frombuffer(audio_bytes, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).copy()


def _float32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] audio to PCM16 bytes."""
    if x is None or len(x) == 0:
        return b""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    return pcm.tobytes()


class STTHandler(AsyncEventHandler):
    def __init__(self, reader, writer):
        super().__init__(reader, writer)
        self.audio_buffer = bytearray()
        self.fmt = AudioFormat()
        self._armed = False  # becomes True after we get "transcribe"

    async def handle_event(self, event: Event) -> bool:
        try:
            # Client capability discovery
            if event.type == "describe":
                await self.write_event(
                    Event(
                        type="info",
                        data={
                            "asr": [
                                {
                                    "name": "mlx-whisper",
                                    "installed": True,
                                    "models": [
                                        {
                                            "name": WHISPER_MODEL_ID.split("/", 1)[-1],
                                            "languages": ["en"],
                                        }
                                    ],
                                }
                            ],
                            "version": "1.6.0",
                        },
                    )
                )
                return False

            # Transcription request (client will follow with audio-start/chunks/stop)
            if event.type == "transcribe":
                self._armed = True
                self.audio_buffer.clear()
                self.fmt = AudioFormat()  # reset to defaults until audio-start arrives
                return True

            # Audio stream start
            if event.type == "audio-start":
                a = AudioStart.from_event(event)
                self.fmt = AudioFormat(rate=a.rate, width=a.width, channels=a.channels)
                self.audio_buffer.clear()
                return True

            # Audio chunk
            if event.type == "audio-chunk":
                if not self._armed:
                    # Ignore stray audio if no transcribe request
                    return True
                chunk = AudioChunk.from_event(event)
                self.audio_buffer.extend(chunk.audio)
                return True

            # Audio stream stop -> run ASR and return transcript
            if event.type == "audio-stop":
                if not self._armed:
                    await self.write_event(Transcript(text="").event())
                    return False

                # Only PCM16 is supported by this drop-in
                if self.fmt.width != 2 or self.fmt.channels != 1:
                    await self.write_event(Transcript(text="").event())
                    self._armed = False
                    self.audio_buffer.clear()
                    return False

                audio_fp32 = _pcm16_bytes_to_float32(bytes(self.audio_buffer))

                # If rate != 16k, mlx-whisper will internally resample if supported;
                # if not, you can add explicit resampling later.
                # Keeping minimal: pass raw float32.
                text = ""
                try:
                    result = mlx_whisper.transcribe(
                        audio_fp32,
                        path_or_hf_repo=WHISPER_MODEL_ID,
                    )
                    text = (result.get("text") or "").strip()
                except Exception:
                    # Never crash the wyoming handler task; return empty transcript
                    traceback.print_exc()
                    text = ""

                await self.write_event(Transcript(text=text).event())

                self._armed = False
                self.audio_buffer.clear()
                return False

            return True

        except Exception:
            traceback.print_exc()
            # Fail closed without killing the server
            try:
                await self.write_event(Transcript(text="").event())
            except Exception:
                pass
            self._armed = False
            self.audio_buffer.clear()
            return False


class TTSHandler(AsyncEventHandler):
    async def handle_event(self, event: Event) -> bool:
        global KOKORO_MODEL

        try:
            if event.type == "describe":
                await self.write_event(
                    Event(
                        type="info",
                        data={
                            "tts": [
                                {
                                    "name": "kokoro-mlx",
                                    "installed": True,
                                    "models": [{"name": "af_heart", "languages": ["en"]}],
                                }
                            ],
                            "version": "1.6.0",
                        },
                    )
                )
                return False

            if event.type != "synthesize":
                return True

            synth = Synthesize.from_event(event)
            text = (synth.text or "").strip()

            # Start audio stream
            await self.write_event(AudioStart(rate=TTS_RATE, width=TTS_WIDTH, channels=TTS_CHANNELS).event())

            if not text:
                await self.write_event(AudioStop().event())
                return False

            try:
                # Kokoro generator yields chunks with .audio float32 array
                for chunk in KOKORO_MODEL.generate(text, voice="af_heart"):
                    audio_pcm = _float32_to_pcm16_bytes(np.array(chunk.audio, dtype=np.float32))
                    if audio_pcm:
                        await self.write_event(
                            AudioChunk(audio=audio_pcm, rate=TTS_RATE, width=TTS_WIDTH, channels=TTS_CHANNELS).event()
                        )
            except Exception:
                traceback.print_exc()

            await self.write_event(AudioStop().event())
            return False

        except Exception:
            traceback.print_exc()
            # Best-effort close stream
            try:
                await self.write_event(AudioStop().event())
            except Exception:
                pass
            return False


async def _warmup_whisper() -> None:
    """
    Force model resolution/download at startup so first real request doesn't crash the handler.
    Uses a short silence buffer.
    """
    silence = np.zeros((DEFAULT_STT_RATE // 2,), dtype=np.float32)  # 0.5s
    try:
        _ = mlx_whisper.transcribe(silence, path_or_hf_repo=WHISPER_MODEL_ID)
    except Exception:
        # Print and continue; server can still run, but STT may fail until corrected
        traceback.print_exc()


async def run_servers() -> None:
    global KOKORO_MODEL

    print(f"🚀 Loading Kokoro ({KOKORO_MODEL_ID})...")
    KOKORO_MODEL = load_kokoro(KOKORO_MODEL_ID)

    print(f"🚀 Warming up Whisper ({WHISPER_MODEL_ID})...")
    await _warmup_whisper()

    stt_server = AsyncServer.from_uri(STT_BIND)
    tts_server = AsyncServer.from_uri(TTS_BIND)

    print("\n✅ SERVER READY")
    print("👉 STT (Whisper) listening on 0.0.0.0:10900")
    print("👉 TTS (Kokoro) listening on 0.0.0.0:10800\n")

    await asyncio.gather(
        stt_server.run(STTHandler),
        tts_server.run(TTSHandler),
    )


if __name__ == "__main__":
    try:
        asyncio.run(run_servers())
    except KeyboardInterrupt:
        print("\nStopping...")
