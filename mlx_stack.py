#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "wyoming==1.6.0",
#   "mlx-whisper==0.4.1",
#   "mlx-audio",
#   "numpy<2",
# ]
# ///

import asyncio
import logging
import re
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

# --- Configuration ---
WHISPER_MODEL_ID = "mlx-community/whisper-turbo"
KOKORO_MODEL_ID = "mlx-community/Kokoro-82M-bf16"

STT_BIND = "tcp://0.0.0.0:10900"
TTS_BIND = "tcp://0.0.0.0:10800"

STT_TARGET_RATE = 16000  
TTS_RATE = 24000
TTS_WIDTH = 2
TTS_CHANNELS = 1

logging.getLogger("phonemizer").setLevel(logging.ERROR)
KOKORO_MODEL = None

@dataclass
class AudioFormat:
    rate: int = STT_TARGET_RATE
    width: int = 2
    channels: int = 1

def normalize_tts_text(s: str) -> str:
    if not s:
        return ""
    # Standardizing characters for MLX processing
    s = (
        s.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2014", "-")
        .replace("\u2013", "-")
    )
    return re.sub(r"\s+", " ", s).strip()

def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.zeros((0,), dtype=np.float32)
    pcm = np.frombuffer(audio_bytes, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).copy()

def float32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    if x is None or len(x) == 0:
        return b""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()

def resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if x.size == 0 or src_rate == dst_rate:
        return x.astype(np.float32, copy=False)
    n_src = x.shape[0]
    n_dst = int(round(n_src * (dst_rate / float(src_rate))))
    if n_dst <= 1:
        return np.zeros((0,), dtype=np.float32)
    src_idx = np.linspace(0.0, n_src - 1, num=n_dst, dtype=np.float32)
    i0 = np.floor(src_idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_src - 1)
    frac = src_idx - i0
    return ((1.0 - frac) * x[i0] + frac * x[i1]).astype(np.float32)

class STTHandler(AsyncEventHandler):
    def __init__(self, reader, writer):
        super().__init__(reader, writer)
        self.audio_buffer = bytearray()
        self.fmt = AudioFormat()
        self._armed = False

    async def handle_event(self, event: Event) -> bool:
        try:
            if event.type == "describe":
                await self.write_event(Event(type="info", data={
                    "asr": [{"name": "mlx-whisper", "installed": True, "models": [{"name": "turbo", "languages": ["en"]}]}],
                    "version": "1.6.0"
                }))
                return False

            if event.type == "transcribe":
                self._armed = True
                self.audio_buffer.clear()
                return True

            if event.type == "audio-start":
                a = AudioStart.from_event(event)
                self.fmt = AudioFormat(rate=a.rate, width=a.width, channels=a.channels)
                return True

            if event.type == "audio-chunk":
                if self._armed:
                    self.audio_buffer.extend(AudioChunk.from_event(event).audio)
                return True

            if event.type == "audio-stop":
                if not self._armed:
                    await self.write_event(Transcript(text="").event())
                    return False

                audio = pcm16_bytes_to_float32(bytes(self.audio_buffer))
                if self.fmt.rate != STT_TARGET_RATE:
                    audio = resample_linear(audio, self.fmt.rate, STT_TARGET_RATE)

                r = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL_ID)
                text = (r.get("text") or "").strip()
                await self.write_event(Transcript(text=text).event())
                self._armed = False
                self.audio_buffer.clear()
                return False
            return True
        except Exception:
            traceback.print_exc()
            return False

class TTSHandler(AsyncEventHandler):
    async def handle_event(self, event: Event) -> bool:
        global KOKORO_MODEL
        try:
            if event.type == "describe":
                # Crucial: Home Assistant looks for this flag to set stream_response: true
                await self.write_event(Event(type="info", data={
                    "tts": [{
                        "name": "kokoro-mlx",
                        "installed": True,
                        "models": [{"name": "af_heart", "languages": ["en"]}],
                        "supports_synthesize_streaming": True
                    }],
                    "version": "1.6.0"
                }))
                return False

            if event.type != "synthesize":
                return True

            synth = Synthesize.from_event(event)
            text = normalize_tts_text((synth.text or "").strip())

            # Signaling start of audio stream
            await self.write_event(AudioStart(rate=TTS_RATE, width=TTS_WIDTH, channels=TTS_CHANNELS).event())

            if text and KOKORO_MODEL:
                # Iterate through model-generated chunks
                for chunk in KOKORO_MODEL.generate(text, voice="af_heart"):
                    pcm = float32_to_pcm16_bytes(np.asarray(chunk.audio, dtype=np.float32))
                    if pcm:
                        await self.write_event(
                            AudioChunk(audio=pcm, rate=TTS_RATE, width=TTS_WIDTH, channels=TTS_CHANNELS).event()
                        )
            
            # Signaling end of audio stream
            await self.write_event(AudioStop().event())
            return False

        except Exception:
            traceback.print_exc()
            try:
                await self.write_event(AudioStop().event())
            except:
                pass
            return False

async def run_servers() -> None:
    global KOKORO_MODEL
    print(f"Loading Kokoro...")
    KOKORO_MODEL = load_kokoro(KOKORO_MODEL_ID)
    
    print("Warming up Whisper...")
    # Pre-load/jit Whisper with silence
    mlx_whisper.transcribe(np.zeros((STT_TARGET_RATE,), dtype=np.float32), path_or_hf_repo=WHISPER_MODEL_ID)
    
    stt_server = AsyncServer.from_uri(STT_BIND)
    tts_server = AsyncServer.from_uri(TTS_BIND)
    print("SERVERS READY - STREAMING ENABLED")
    
    await asyncio.gather(
        stt_server.run(STTHandler),
        tts_server.run(TTSHandler),
    )

if __name__ == "__main__":
    asyncio.run(run_servers())