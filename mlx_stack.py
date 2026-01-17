#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "wyoming>=1.8.0",
#   "mlx-whisper==0.4.1",
#   "mlx-audio",
#   "numpy<2",
#   "httpx",
# ]
# ///

import asyncio
import logging
import numpy as np
import mlx_whisper
import httpx
from mlx_audio.tts.utils import load_model as load_kokoro

from wyoming.event import Event
from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.asr import Transcript
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.tts import Synthesize, SynthesizeStart, SynthesizeChunk, SynthesizeStop, SynthesizeStopped
from wyoming.info import AsrModel, AsrProgram, Info, TtsVoice, TtsProgram, Attribution

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOGGER = logging.getLogger("mlx_stack")

# --- Config ---
WHISPER_MODEL_ID = "mlx-community/whisper-turbo"
KOKORO_MODEL_ID = "mlx-community/Kokoro-82M-bf16"

# Verified Repo and File casing for Qwen 2.5 7B
LLM_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
LLM_FILE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

STT_BIND = "tcp://0.0.0.0:10900"
TTS_BIND = "tcp://0.0.0.0:10800"
LLM_HOST = "0.0.0.0"
LLM_PORT = 8033
TTS_RATE = 24000
KOKORO_MODEL = None

KOKORO_VOICES = [
    ("af_heart", "en-US", "Heart ❤️"), ("af_alloy", "en-US", "Alloy"), ("af_aoede", "en-US", "Aoede"),
    ("af_bella", "en-US", "Bella 🔥"), ("af_jessica", "en-US", "Jessica"), ("af_kore", "en-US", "Kore"),
    ("af_nicole", "en-US", "Nicole"), ("af_nova", "en-US", "Nova"), ("af_river", "en-US", "River"),
    ("af_sarah", "en-US", "Sarah"), ("af_sky", "en-US", "Sky"), ("am_adam", "en-US", "Adam"),
    ("am_echo", "en-US", "Echo"), ("am_eric", "en-US", "Eric"), ("am_fenrir", "en-US", "Fenrir"),
    ("am_liam", "en-US", "Liam"), ("am_michael", "en-US", "Michael"), ("am_onyx", "en-US", "Onyx"),
    ("am_puck", "en-US", "Puck"), ("am_santa", "en-US", "Santa 🎅"),
    ("bf_alice", "en-GB", "Alice"), ("bf_emma", "en-GB", "Emma"), ("bf_isabella", "en-GB", "Isabella"),
    ("bf_lily", "en-GB", "Lily"), ("bm_daniel", "en-GB", "Daniel"), ("bm_fable", "en-GB", "Fable"),
    ("bm_george", "en-GB", "George"), ("bm_lewis", "en-GB", "Lewis"),
    ("jf_alpha", "ja", "Alpha (JA)"), ("zf_xiaoxiao", "zh", "Xiaoxiao (ZH)"),
    ("ef_dora", "es", "Dora (ES)"), ("em_alex", "es", "Alex (ES)"),
    ("ff_siwis", "fr", "Siwis (FR)"), ("hf_alpha", "hi", "Alpha (HI)"),
    ("if_sara", "it", "Sara (IT)"), ("pf_dora", "pt-BR", "Dora (PT)")
]

async def start_llama_server():
    cmd = [
        "llama-server", 
        "-fa", "on", 
        "-hf", LLM_REPO, 
        "--hf-file", LLM_FILE,
        "--host", LLM_HOST, 
        "--port", str(LLM_PORT),
        "--jinja",
        "-c", "8192",
        "--context-shift"       # Prevents crashes when conversation gets long
    ]
    
    _LOGGER.info(f"🧠 Starting LLM Server: {LLM_FILE}")
    # stdout=None so you can see the download progress bar in your terminal
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=None, stderr=None)
    
    _LOGGER.info("⏳ Waiting for LLM (Watch terminal for download progress)...")
    async with httpx.AsyncClient() as client:
        # 10 minute timeout to ensure the first download completes
        for i in range(300): 
            try:
                resp = await client.get(f"http://localhost:{LLM_PORT}/health")
                if resp.status_code == 200:
                    _LOGGER.info(f"✅ LLM ({LLM_FILE}) is available.")
                    return proc
            except Exception:
                pass
            await asyncio.sleep(2)
            
    _LOGGER.error("❌ LLM failed to start after 10 minutes.")
    return proc

class STTHandler(AsyncEventHandler):
    def __init__(self, reader, writer):
        super().__init__(reader, writer)
        self.audio_buffer = bytearray()
        self._armed = False

    async def handle_event(self, event: Event) -> bool:
        if event.type == "describe":
            attribution = Attribution(name="OpenAI", url="https://github.com/openai/whisper")
            info = Info(asr=[AsrProgram(
                name="mlx-whisper", 
                description="MLX Whisper",
                attribution=attribution,
                installed=True, 
                version="1.8.3",
                models=[AsrModel(
                    name=WHISPER_MODEL_ID, 
                    description="Whisper Turbo", 
                    attribution=attribution,
                    installed=True, 
                    version="turbo",
                    languages=["en"]
                )]
            )])
            await self.write_event(info.event())
            return False
        if event.type == "transcribe":
            self._armed = True
            self.audio_buffer.clear()
            return True
        if event.type == "audio-chunk" and self._armed:
            self.audio_buffer.extend(AudioChunk.from_event(event).audio)
            return True
        if event.type == "audio-stop":
            audio = (np.frombuffer(bytes(self.audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0)
            loop = asyncio.get_running_loop()
            r = await loop.run_in_executor(None, lambda: mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL_ID))
            text = (r.get("text") or "").strip()
            if text: _LOGGER.info(f"📝 User: {text}")
            await self.write_event(Transcript(text=text).event())
            self._armed = False
            return False
        return True

class TTSHandler(AsyncEventHandler):
    def __init__(self, reader, writer):
        super().__init__(reader, writer)
        self._streaming_active = False
        self._voice_name = "af_sky"
        self._audio_started = False
        self._text_buffer = []

    async def _ensure_audio_start(self):
        if not self._audio_started:
            await self.write_event(AudioStart(rate=TTS_RATE, width=2, channels=1).event())
            self._audio_started = True

    async def _synthesize_text(self, text: str, voice: str, mode: str):
        global KOKORO_MODEL
        if not text or not KOKORO_MODEL:
            return
        _LOGGER.info(f"🔊 AI ({mode}): {text[:100]}...")
        await self._ensure_audio_start()
        loop = asyncio.get_running_loop()
        def generate_pcm():
            for chunk in KOKORO_MODEL.generate(text, voice=voice):
                yield (np.clip(chunk.audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        chunks = await loop.run_in_executor(None, lambda: list(generate_pcm()))
        for pcm_data in chunks:
            await self.write_event(AudioChunk(audio=pcm_data, rate=TTS_RATE, width=2, channels=1).event())

    def _should_flush_buffer(self, force: bool) -> bool:
        if not self._text_buffer: return False
        if force: return True
        text = "".join(self._text_buffer).rstrip()
        if len(text) >= 300: return True
        return text.endswith((".", "!", "?", "\n"))

    async def _flush_text_buffer(self, force: bool = False):
        if not self._should_flush_buffer(force): return
        text = "".join(self._text_buffer)
        self._text_buffer = []
        await self._synthesize_text(text, self._voice_name, "stream")

    async def handle_event(self, event: Event) -> bool:
        global KOKORO_MODEL
        if event.type == "describe":
            attribution = Attribution(name="Kokoro TTS", url="https://github.com/hexgrad/Kokoro-82M")
            wy_voices = [TtsVoice(name=v[0], description=v[2], attribution=attribution, installed=True, version="82M", languages=[v[1]]) for v in KOKORO_VOICES]
            info = Info(tts=[TtsProgram(name="kokoro-mlx", description="Kokoro TTS", attribution=attribution, installed=True, version="1.8.3", voices=wy_voices, supports_synthesize_streaming=True)])
            await self.write_event(info.event())
            return False
        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            self._streaming_active, self._audio_started = True, False
            self._text_buffer = []
            self._voice_name = start.voice.name if (start.voice and start.voice.name) else "af_sky"
            return True
        if SynthesizeChunk.is_type(event.type):
            self._text_buffer.append(SynthesizeChunk.from_event(event).text)
            await self._flush_text_buffer(force=False)
            return True
        if SynthesizeStop.is_type(event.type):
            await self._flush_text_buffer(force=True)
            if self._audio_started: await self.write_event(AudioStop().event())
            await self.write_event(SynthesizeStopped().event())
            self._streaming_active = False
            return True
        if Synthesize.is_type(event.type):
            if self._streaming_active: return True
            synth = Synthesize.from_event(event)
            self._audio_started = False
            await self._synthesize_text((synth.text or "").strip(), synth.voice.name if (synth.voice and synth.voice.name) else "af_sky", "legacy")
            await self.write_event(AudioStop().event())
            return False
        return True

async def run_servers() -> None:
    global KOKORO_MODEL
    llm_proc = await start_llama_server()
    _LOGGER.info("🔊 Loading Kokoro...")
    KOKORO_MODEL = load_kokoro(KOKORO_MODEL_ID)
    stt_server, tts_server = AsyncServer.from_uri(STT_BIND), AsyncServer.from_uri(TTS_BIND)
    _LOGGER.info(f"🚀 STT @ {STT_BIND} | TTS @ {TTS_BIND}")
    try:
        await asyncio.gather(stt_server.run(STTHandler), tts_server.run(TTSHandler))
    finally:
        if llm_proc:
            llm_proc.terminate()
            await llm_proc.wait()

if __name__ == "__main__":
    try:
        asyncio.run(run_servers())
    except KeyboardInterrupt:
        pass