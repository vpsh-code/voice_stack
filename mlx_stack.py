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
import os
import re
import signal
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List

import httpx
import mlx_whisper
import numpy as np
from mlx_audio.tts.utils import load_model as load_kokoro

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

# ==============================================================================
# LOGGING (easy on the eye)
# ==============================================================================

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOGGER = logging.getLogger("mlx_stack")

for noisy in ("httpcore", "httpx", "mlx_audio", "mlx_whisper", "wyoming.server"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ==============================================================================
# CONFIG
# ==============================================================================

WHISPER_MODEL_ID = "mlx-community/whisper-turbo"
KOKORO_MODEL_ID = "mlx-community/Kokoro-82M-bf16"

LLM_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
LLM_FILE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
LLM_HOST = "0.0.0.0"
LLM_PORT = 8033

STT_BIND = "tcp://0.0.0.0:10900"
TTS_BIND = "tcp://0.0.0.0:10800"
TTS_RATE = 24000

# Ensure MAX context window (matches model context_length=32768)
LLM_CTX = int(os.environ.get("LLM_CTX", "32768"))

# Prompt cache: keep enabled by default (helps HA responsiveness). Set 0 to disable.
LLM_CACHE_RAM = os.environ.get("LLM_CACHE_RAM", "4096").strip()

# Try fast-start flags; if unsupported, auto-fallback
LLM_FLAG_SETS = [
    ["--no-warmup"],
    [],
]

# Logging controls
PASSTHRU_LLM_LOGS = os.environ.get("LLM_LOG_PASSTHRU", "").strip() == "1"
LLM_LOG_FILE = os.environ.get(
    "LLM_LOG_FILE",
    f"/tmp/llama-server-{time.strftime('%Y%m%d-%H%M%S')}.log",
)

# TTS warmup behavior:
# - "0" => do not warm (fast startup)
# - "bg" => warm in background (recommended, no startup regression)
# - "1" => warm synchronously (slow startup, best first-turn)
KOKORO_PREWARM = os.environ.get("KOKORO_PREWARM", "bg").strip().lower()

# Kokoro language pinning (prevents accidental pipeline creation for nonsense lang like "a")
KOKORO_LANG = os.environ.get("KOKORO_LANG", "en").strip().lower()

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
    ("if_sara", "it", "Sara (IT)"), ("pf_dora", "pt-BR", "Dora (PT)"),
]

# ==============================================================================
# TELEMETRY
# ==============================================================================

@dataclass
class StackTelemetry:
    t0: float = field(default_factory=time.time)

    llm_ready_at: Optional[float] = None
    tts_ready_at: Optional[float] = None
    wyoming_serving_at: Optional[float] = None

    # Latency markers (first-ever in-process; per-turn requires task-scoped state)
    first_token_at: Optional[float] = None
    first_voice_at: Optional[float] = None

    stt_count: int = 0
    tts_count: int = 0
    llm_req_count: int = 0

    def _since0(self, t: Optional[float]) -> str:
        if t is None:
            return "-"
        return f"{(t - self.t0):.2f}s"

    def banner(self):
        _LOGGER.info("")
        _LOGGER.info("──────────── Stack Telemetry ────────────")
        _LOGGER.info("Shows: FULL stack ready time, per-response timings, context size, and context shift")
        _LOGGER.info("Legend:")
        _LOGGER.info("  ctx   = prompt tokens used / window (n_ctx)")
        _LOGGER.info("  shift = window slides; oldest tokens are dropped to fit new prompt (log: memory_seq_rm)")
        _LOGGER.info("  offload(model) = weights/layers placed on GPU (Metal); separate from context shift")
        _LOGGER.info("─────────────────────────────────────────")
        _LOGGER.info("")

    def log_ttft(self):
        if self.first_token_at is not None:
            _LOGGER.info(f"[LATENCY] TTFT {self._since0(self.first_token_at)}")

    def log_ttfv(self):
        if self.first_voice_at is not None:
            _LOGGER.info(f"[LATENCY] TTFV {self._since0(self.first_voice_at)}")

    def log_full_ready(self):
        if self.llm_ready_at and self.tts_ready_at and self.wyoming_serving_at:
            full = max(self.llm_ready_at, self.tts_ready_at, self.wyoming_serving_at)
            _LOGGER.info(
                f"[STACK] FULL ready @ {self._since0(full)} "
                f"(LLM {self._since0(self.llm_ready_at)}, TTS {self._since0(self.tts_ready_at)}, Wyoming {self._since0(self.wyoming_serving_at)})"
            )

    def log_stt(self, seconds: float, text: str):
        self.stt_count += 1
        preview = text.replace("\n", " ").strip()
        if len(preview) > 90:
            preview = preview[:90] + "…"
        _LOGGER.info(f"[STT #{self.stt_count:>3}] {seconds:.2f}s | \"{preview}\"")

    def log_tts(self, seconds: float, chars: int, audio_ms: Optional[int], voice: str, mode: str):
        self.tts_count += 1
        ams = "-" if audio_ms is None else f"{audio_ms}ms"
        _LOGGER.info(
            f"[TTS #{self.tts_count:>3}] {mode:<6} {seconds:.2f}s | chars {chars:>4} | audio {ams:>7} | voice {voice}"
        )

    def log_llm_summary(
        self,
        task_id: int,
        n_ctx_slot: Optional[int],
        task_tokens: Optional[int],
        prompt_ms: Optional[float],
        prompt_tok: Optional[int],
        gen_ms: Optional[float],
        gen_tok: Optional[int],
        total_ms: Optional[float],
        shifted_from: Optional[int],
    ):
        self.llm_req_count += 1

        ctx_part = "-"
        if n_ctx_slot and task_tokens is not None:
            pct = (task_tokens / n_ctx_slot) * 100.0
            ctx_part = f"{task_tokens}/{n_ctx_slot} ({pct:>4.1f}%)"

        shift_part = "no"
        if shifted_from is not None:
            shift_part = f"yes (drop >= {shifted_from})"

        def fmt_time(ms: Optional[float]) -> str:
            return "-" if ms is None else f"{ms/1000.0:.2f}s"

        def fmt_tok(n: Optional[int]) -> str:
            return "-" if n is None else str(n)

        _LOGGER.info(
            f"[LLM #{self.llm_req_count:>3}] task {task_id:<4} | total {fmt_time(total_ms):>6} "
            f"| prompt {fmt_tok(prompt_tok):>4} tok {fmt_time(prompt_ms):>6} "
            f"| gen {fmt_tok(gen_tok):>4} tok {fmt_time(gen_ms):>6} "
            f"| ctx {ctx_part:<18} | shift {shift_part}"
        )


TELEM = StackTelemetry()

# ==============================================================================
# LLM LOG PARSING (from file tail)
#   - No stdout/stderr pipes -> no backpressure -> prevents HA "error talking to API"
#   - We tail the log file in a thread -> never blocks the asyncio event loop
# ==============================================================================

@dataclass
class _LlmTaskState:
    task_id: int
    n_ctx_slot: Optional[int] = None
    task_tokens: Optional[int] = None
    shifted_from: Optional[int] = None
    prompt_ms: Optional[float] = None
    prompt_tok: Optional[int] = None
    gen_ms: Optional[float] = None
    gen_tok: Optional[int] = None
    total_ms: Optional[float] = None

class LlamaLogParser:
    RE_NEW_PROMPT = re.compile(
        r"slot update_slots: id\s+\d+\s+\|\s+task\s+(\d+)\s+\|.*new prompt, n_ctx_slot\s+=\s+(\d+).*task\.n_tokens\s+=\s+(\d+)"
    )
    RE_SEQ_RM = re.compile(r"memory_seq_rm\s+\[(\d+),\s*end\)")
    RE_PROMPT = re.compile(r"prompt eval time\s+=\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens")
    RE_GEN = re.compile(r"^\s*eval time\s+=\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens")
    RE_TOTAL = re.compile(r"total time\s+=\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens")

    def __init__(self):
        self.current_task: Optional[_LlmTaskState] = None
        self.last_shift_boundary: Optional[int] = None
        self._printed_intro = False
        self._offload_layers: Optional[str] = None
        self._kv_buf: Optional[str] = None

    def feed(self, line: str):
        if PASSTHRU_LLM_LOGS:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

        if not self._printed_intro:
            m = re.search(r"offloaded\s+(\d+/\d+)\s+layers to GPU", line)
            if m:
                self._offload_layers = m.group(1)

            m = re.search(r"Metal KV buffer size\s+=\s+([0-9.]+\s+MiB)", line)
            if m:
                self._kv_buf = m.group(1)

            if (self._offload_layers and self._kv_buf) or ("main: model loaded" in line):
                _LOGGER.info(f"[LLM] offload(model): {self._offload_layers or '?'} layers on Metal")
                _LOGGER.info(f"[LLM] context memory: KV cache ~ {self._kv_buf or '?'} (scales with n_ctx={LLM_CTX})")
                _LOGGER.info("[LLM] shift(context): enabled; if prompt exceeds n_ctx, oldest tokens are dropped (memory_seq_rm)")
                _LOGGER.info(f"[LLM] logs: {LLM_LOG_FILE}")
                self._printed_intro = True

        m = self.RE_SEQ_RM.search(line)
        if m:
            self.last_shift_boundary = int(m.group(1))
            return

        m = self.RE_NEW_PROMPT.search(line)
        if m:
            task_id = int(m.group(1))
            n_ctx = int(m.group(2))
            n_tok = int(m.group(3))
            self.current_task = _LlmTaskState(
                task_id=task_id,
                n_ctx_slot=n_ctx,
                task_tokens=n_tok,
                shifted_from=self.last_shift_boundary,
            )
            self.last_shift_boundary = None
            return

        if not self.current_task:
            return

        m = self.RE_PROMPT.search(line)
        if m:
            self.current_task.prompt_ms = float(m.group(1))
            self.current_task.prompt_tok = int(m.group(2))
            return

        m = self.RE_GEN.search(line)
        if m:
            self.current_task.gen_ms = float(m.group(1))
            self.current_task.gen_tok = int(m.group(2))

            # First token observed (global, in-process)
            if TELEM.first_token_at is None:
                TELEM.first_token_at = time.time()
                TELEM.log_ttft()

            return

        m = self.RE_TOTAL.search(line)
        if m:
            self.current_task.total_ms = float(m.group(1))
            TELEM.log_llm_summary(
                task_id=self.current_task.task_id,
                n_ctx_slot=self.current_task.n_ctx_slot,
                task_tokens=self.current_task.task_tokens,
                prompt_ms=self.current_task.prompt_ms,
                prompt_tok=self.current_task.prompt_tok,
                gen_ms=self.current_task.gen_ms,
                gen_tok=self.current_task.gen_tok,
                total_ms=self.current_task.total_ms,
                shifted_from=self.current_task.shifted_from,
            )
            return

def _tail_file_in_thread(path: str, parser: LlamaLogParser, stop_evt: threading.Event):
    def run():
        while not stop_evt.is_set():
            if os.path.exists(path):
                break
            time.sleep(0.05)

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                while not stop_evt.is_set():
                    line = f.readline()
                    if line:
                        parser.feed(line.rstrip("\n"))
                        continue
                    time.sleep(0.05)
        except Exception:
            pass

    t = threading.Thread(target=run, name="llm-log-tail", daemon=True)
    t.start()
    return t

# ==============================================================================
# LLM STARTUP
# ==============================================================================

async def _llm_is_ready(client: httpx.AsyncClient) -> bool:
    for path in ("/health", "/healthz", "/v1/models"):
        try:
            r = await client.get(path, timeout=0.35)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False

def _build_llama_cmd(extra_flags: List[str]) -> List[str]:
    cmd = [
        "llama-server",
        "-fa", "on",
        "-hf", LLM_REPO,
        "--hf-file", LLM_FILE,
        "--host", LLM_HOST,
        "--port", str(LLM_PORT),
        "--jinja",
        "-c", str(LLM_CTX),
        "--context-shift",
        *extra_flags,
    ]
    if LLM_CACHE_RAM and LLM_CACHE_RAM != "0":
        cmd += ["--cache-ram", LLM_CACHE_RAM]
    return cmd

async def start_llama_server_nonblocking(llm_ready_evt: asyncio.Event):
    parser = LlamaLogParser()
    stop_evt = threading.Event()
    _tail_file_in_thread(LLM_LOG_FILE, parser, stop_evt)

    os.makedirs(os.path.dirname(LLM_LOG_FILE), exist_ok=True)
    log_fp = open(LLM_LOG_FILE, "ab", buffering=0)

    last_proc = None
    for flags in LLM_FLAG_SETS:
        cmd = _build_llama_cmd(flags)
        _LOGGER.info(f"🧠 Starting LLM Server: {LLM_FILE} {' '.join(flags) if flags else ''}".rstrip())
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=log_fp, stderr=log_fp)

        await asyncio.sleep(0.8)
        if proc.returncode is None:
            last_proc = proc
            break

        last_proc = proc
        _LOGGER.warning(f"[LLM] exited early (rc={proc.returncode}) using flags: {flags}. Falling back...")

    async def watcher():
        if not last_proc:
            _LOGGER.error("❌ LLM: failed to spawn llama-server")
            return False

        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{LLM_PORT}") as client:
            while True:
                if last_proc.returncode is not None:
                    _LOGGER.error(f"❌ LLM: process exited (rc={last_proc.returncode})")
                    return False

                if await _llm_is_ready(client):
                    if TELEM.llm_ready_at is None:
                        TELEM.llm_ready_at = time.time()
                        llm_ready_evt.set()
                        _LOGGER.info(f"[LLM] ready @ {TELEM._since0(TELEM.llm_ready_at)} (health=200)")
                    return True

                await asyncio.sleep(0.25)

    try:
        log_fp.close()
    except Exception:
        pass

    return last_proc, asyncio.create_task(watcher()), stop_evt

# ==============================================================================
# STT / TTS
# ==============================================================================

def _clean_tts_text(text: str) -> str:
    t = (text or "").replace("\r", "").strip()
    if not t:
        return ""
    t = re.sub(r"^\s*[-*]\s+", "", t, flags=re.M)
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

async def _prewarm_kokoro():
    """Build Kokoro pipeline and generate a tiny amount of audio.
    Run this in background by default to avoid startup regressions.
    """
    global KOKORO_MODEL
    if not KOKORO_MODEL:
        return
    loop = asyncio.get_running_loop()

    def _warm():
        try:
            # language pinned to prevent accidental pipeline variants
            for _ in KOKORO_MODEL.generate("Okay.", voice="af_sky", lang=KOKORO_LANG):
                break
        except Exception:
            pass

    await loop.run_in_executor(None, _warm)

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
                    languages=["en"],
                )],
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

            t0 = time.time()
            r = await loop.run_in_executor(None, lambda: mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL_ID))
            dt = time.time() - t0

            text = (r.get("text") or "").strip()
            if text:
                TELEM.log_stt(seconds=dt, text=text)

            await self.write_event(Transcript(text=text).event())
            self._armed = False
            return False

        return True

# ------------------------------------------------------------------------------
# Speech Commit Controller (Pipecat/LiveKit-style)
#   - Stability window: do not speak while text is still changing
#   - Minimum chunk: ensure natural prosody (avoid too-short fragments)
#   - Boundary confidence: punctuation helps, but time+stability can also trigger
#   - Numeric continuity guard: prevents flushing inside "5,000" / "3.14"
# ------------------------------------------------------------------------------

_NUMERIC_CONTINUATION_RE = re.compile(r"(\d[,\.\s]?)$")  # minimal guard, not a rulebook

def _ends_like_number_in_progress(s: str) -> bool:
    s = s.rstrip()
    if not s:
        return False
    # If tail is digit/comma/dot, likely still forming a number token group
    # (stability gate handles most cases; this prevents comma-triggered early flush)
    return bool(_NUMERIC_CONTINUATION_RE.search(s)) and any(ch.isdigit() for ch in s[-4:])

class SpeechCommitController:
    def __init__(self):
        # Tunables chosen to improve naturalness without adding audible latency
        self.min_stable_ms = int(os.environ.get("TTS_STABLE_MS", "140"))          # stability window
        self.min_words_first = int(os.environ.get("TTS_MIN_WORDS_FIRST", "6"))   # first chunk
        self.min_words_next = int(os.environ.get("TTS_MIN_WORDS_NEXT", "3"))     # follow chunks
        self.max_chars_hold = int(os.environ.get("TTS_MAX_CHARS_HOLD", "260"))   # safety release
        self.min_commit_delay_ms = int(os.environ.get("TTS_MIN_COMMIT_DELAY_MS", "220"))  # delayed commitment

        # internal state
        self.buffer: List[str] = []
        self.last_change_at = time.time()
        self.start_at = time.time()

    def reset(self):
        self.buffer = []
        now = time.time()
        self.last_change_at = now
        self.start_at = now

    def append(self, text: str):
        if not text:
            return
        self.buffer.append(text)
        self.last_change_at = time.time()

    def get_text(self) -> str:
        return "".join(self.buffer)

    def should_commit(self, audio_started: bool, force: bool) -> bool:
        if not self.buffer:
            return False

        text = self.get_text()
        clean = text.strip()
        if not clean:
            return False

        if force:
            # even on force, avoid breaking numbers in the middle if we can
            if _ends_like_number_in_progress(clean):
                return False
            return True

        now = time.time()
        stable_ms = int((now - self.last_change_at) * 1000.0)

        # 1) Stability gate: do not speak while text is still changing
        if stable_ms < self.min_stable_ms:
            return False

        # 2) Delayed commitment: tiny intentional hold to improve prosody
        since_start_ms = int((now - self.start_at) * 1000.0)
        if not audio_started and since_start_ms < self.min_commit_delay_ms:
            return False

        # 3) Avoid committing inside a number token/group
        if _ends_like_number_in_progress(clean):
            return False

        # 4) Minimum chunk size
        words = len(clean.split())
        if not audio_started:
            if words < self.min_words_first:
                # allow commit if we have a strong boundary anyway
                if clean.endswith((".", "!", "?", "\n")):
                    return True
                return False
        else:
            if words < self.min_words_next:
                if clean.endswith((".", "!", "?", "\n")):
                    return True
                return False

        # 5) Boundary confidence
        if clean.endswith((".", "!", "?", "\n")):
            return True

        # Allow comma/semicolon only if chunk is already "speech-sized"
        if clean.endswith((",", ";", ":")) and words >= (self.min_words_first if not audio_started else self.min_words_next) + 2:
            return True

        # 6) Safety release: don't hold too long (keeps responsiveness)
        if len(clean) >= self.max_chars_hold:
            return True

        return False

    def pop_commit_text(self) -> str:
        text = self.get_text()
        self.buffer = []
        self.start_at = time.time()
        return text

class TTSHandler(AsyncEventHandler):
    def __init__(self, reader, writer):
        super().__init__(reader, writer)
        self._streaming_active = False
        self._voice_name = "af_sky"
        self._audio_started = False
        self._first_audio_sent = False

        # controller replaces naive punctuation/length flush
        self._commit = SpeechCommitController()

        # streaming audio bridge (thread -> asyncio queue)
        self._pcm_queue: Optional[asyncio.Queue] = None
        self._pcm_worker_task: Optional[asyncio.Task] = None

    async def _ensure_audio_start(self):
        if not self._audio_started:
            await self.write_event(AudioStart(rate=TTS_RATE, width=2, channels=1).event())
            self._audio_started = True

    async def _pcm_writer_worker(self, q: asyncio.Queue):
        """Consumes PCM bytes and emits AudioChunk events."""
        while True:
            item = await q.get()
            if item is None:
                return
            pcm_data = item
            if not self._first_audio_sent:
                self._first_audio_sent = True
                if TELEM.first_voice_at is None:
                    TELEM.first_voice_at = time.time()
                    TELEM.log_ttfv()
            await self.write_event(AudioChunk(audio=pcm_data, rate=TTS_RATE, width=2, channels=1).event())

    async def _synthesize_text_streaming(self, text: str, voice: str, mode: str):
        """True streaming: as Kokoro yields frames, we emit AudioChunk with no batching.
        Implementation keeps the event loop safe by producing audio in a thread and pushing PCM to an asyncio queue.
        """
        global KOKORO_MODEL
        clean = _clean_tts_text(text)
        if not clean or not KOKORO_MODEL:
            return

        await self._ensure_audio_start()

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=24)
        self._pcm_queue = q

        if self._pcm_worker_task is None or self._pcm_worker_task.done():
            self._pcm_worker_task = asyncio.create_task(self._pcm_writer_worker(q))

        t0 = time.time()
        total_bytes = 0

        def producer():
            nonlocal total_bytes
            try:
                # language pinned to prevent accidental pipeline variants
                for chunk in KOKORO_MODEL.generate(clean, voice=voice, lang=KOKORO_LANG):
                    pcm = (np.clip(chunk.audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    total_bytes += len(pcm)
                    # backpressure via bounded queue; block in producer thread only
                    asyncio.run_coroutine_threadsafe(q.put(pcm), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), loop).result()

        prod_thread = threading.Thread(target=producer, name="kokoro-producer", daemon=True)
        prod_thread.start()

        # wait for producer completion (thread); audio writer keeps emitting in parallel
        while prod_thread.is_alive():
            await asyncio.sleep(0.02)

        # ensure writer finishes
        if self._pcm_worker_task:
            try:
                await asyncio.wait_for(self._pcm_worker_task, timeout=5.0)
            except Exception:
                pass

        synth_s = time.time() - t0

        audio_ms = None
        try:
            samples = total_bytes // 2
            audio_ms = int((samples / float(TTS_RATE)) * 1000.0)
        except Exception:
            pass

        TELEM.log_tts(seconds=synth_s, chars=len(clean), audio_ms=audio_ms, voice=voice, mode=mode)

    async def _maybe_commit(self, force: bool = False):
        if self._commit.should_commit(audio_started=self._audio_started, force=force):
            text = self._commit.pop_commit_text()
            await self._synthesize_text_streaming(text, self._voice_name, "stream")

    async def handle_event(self, event: Event) -> bool:
        if event.type == "describe":
            attribution = Attribution(name="Kokoro TTS", url="https://github.com/hexgrad/Kokoro-82M")
            wy_voices = [
                TtsVoice(
                    name=v[0],
                    description=v[2],
                    attribution=attribution,
                    installed=True,
                    version="82M",
                    languages=[v[1]],
                )
                for v in KOKORO_VOICES
            ]
            info = Info(tts=[TtsProgram(
                name="kokoro-mlx",
                description="Kokoro TTS",
                attribution=attribution,
                installed=True,
                version="1.8.3",
                voices=wy_voices,
                supports_synthesize_streaming=True,  # protect streaming
            )])
            await self.write_event(info.event())
            return False

        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            self._streaming_active = True
            self._audio_started = False
            self._first_audio_sent = False

            self._commit.reset()
            self._voice_name = start.voice.name if (start.voice and start.voice.name) else "af_sky"
            return True

        if SynthesizeChunk.is_type(event.type):
            chunk = SynthesizeChunk.from_event(event).text
            self._commit.append(chunk)
            await self._maybe_commit(force=False)
            return True

        if SynthesizeStop.is_type(event.type):
            # best-effort: wait a short moment for stability before forcing flush
            await asyncio.sleep(0.06)
            # if still ends in a number fragment, give it another breath to stabilize
            if _ends_like_number_in_progress(self._commit.get_text().strip()):
                await asyncio.sleep(0.08)

            await self._maybe_commit(force=True)

            if self._audio_started:
                await self.write_event(AudioStop().event())
            await self.write_event(SynthesizeStopped().event())
            self._streaming_active = False
            return True

        # Legacy non-streaming synthesize path
        if Synthesize.is_type(event.type):
            if self._streaming_active:
                return True
            synth = Synthesize.from_event(event)
            self._audio_started = False
            self._first_audio_sent = False

            await self._synthesize_text_streaming(
                (synth.text or "").strip(),
                synth.voice.name if (synth.voice and synth.voice.name) else "af_sky",
                "legacy",
            )
            await self.write_event(AudioStop().event())
            return False

        return True

# ==============================================================================
# MAIN
# ==============================================================================

async def run_servers() -> None:
    global KOKORO_MODEL

    TELEM.banner()

    llm_ready_evt = asyncio.Event()
    tts_ready_evt = asyncio.Event()

    llm_proc, llm_watch, llm_tail_stop = await start_llama_server_nonblocking(llm_ready_evt)

    _LOGGER.info("🔊 Loading Kokoro...")
    t0 = time.time()
    KOKORO_MODEL = load_kokoro(KOKORO_MODEL_ID)

    TELEM.tts_ready_at = time.time()
    tts_ready_evt.set()
    _LOGGER.info(f"[TTS] ready @ {TELEM._since0(TELEM.tts_ready_at)} (load {TELEM.tts_ready_at - t0:.2f}s)")

    # Warmup strategy (default: background, avoids startup regression)
    if KOKORO_PREWARM in ("1", "true", "yes"):
        await _prewarm_kokoro()
    elif KOKORO_PREWARM in ("bg", "background"):
        asyncio.create_task(_prewarm_kokoro())
    else:
        pass

    stt_server = AsyncServer.from_uri(STT_BIND)
    tts_server = AsyncServer.from_uri(TTS_BIND)
    TELEM.wyoming_serving_at = time.time()

    async def full_ready_announcer():
        await llm_ready_evt.wait()
        await tts_ready_evt.wait()
        TELEM.log_full_ready()

    asyncio.create_task(full_ready_announcer())

    try:
        await asyncio.gather(stt_server.run(STTHandler), tts_server.run(TTSHandler))
    finally:
        try:
            llm_tail_stop.set()
        except Exception:
            pass

        if llm_watch and not llm_watch.done():
            llm_watch.cancel()
            try:
                await llm_watch
            except Exception:
                pass

        if llm_proc and llm_proc.returncode is None:
            try:
                llm_proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                pass
            try:
                await llm_proc.wait()
            except Exception:
                pass

if __name__ == "__main__":
    try:
        asyncio.run(run_servers())
    except KeyboardInterrupt:
        pass
