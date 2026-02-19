#!/usr/bin/env python3
"""Voice-to-Text: press F8 to start/stop recording, transcribe with AI.

Press F9 to cancel (during recording or transcription).

AUDIO COMPRESSION: Records as WAV, automatically converts to OGG/OPUS for upload
(10-20x smaller files = much faster transmission)

Set hotkey via VOICE2TEXT_HOTKEY env variable (default: f8)
Set cancel key via VOICE2TEXT_CANCEL_KEY env variable (default: f9)
Options: f1-f12, insert (insert only works on external keyboards)

Set model via: export VOICE2TEXT_MODEL="gemini-3-flash"
"""

from __future__ import annotations

# Suppress urllib3 OpenSSL warning BEFORE importing requests/urllib3
import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import argparse
import base64
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
import pyperclip
import requests
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

__all__ = ["main", "App", "Recorder"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables."""

    api_key: str
    api_base_url: str = "https://openrouter.ai/api/v1"
    model_alias: str = "gemini-3-flash"
    hotkey_name: str = "f8"
    cancel_key_name: str = "f9"
    save_dir: Path = Path.cwd()
    sample_rate: int = 16000
    min_duration_sec: float = 0.3
    transcribe_prompt: str = (
        "Transcribe the following audio exactly as spoken. Output only the transcription text, nothing else."
    )
    insert_vk: int = 114

    @property
    def saved_recording_path(self) -> Path:
        return self.save_dir / "recording.wav"

    @property
    def saved_ogg_path(self) -> Path:
        return self.save_dir / "recording.ogg"

    @classmethod
    def from_env(cls) -> Config:
        """Create configuration from environment variables.

        Loads .env file if present (does not override existing env vars).
        """
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        return cls(
            api_key=os.environ.get("VOICE2TEXT_API_KEY", ""),
            api_base_url=os.environ.get("VOICE2TEXT_API_URL", "https://openrouter.ai/api/v1"),
            model_alias=os.environ.get("VOICE2TEXT_MODEL", "gemini-3-flash"),
            hotkey_name=os.environ.get("VOICE2TEXT_HOTKEY", "f8").lower(),
            cancel_key_name=os.environ.get("VOICE2TEXT_CANCEL_KEY", "f9").lower(),
        )

    def validate(self) -> None:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError("API key not set. Set VOICE2TEXT_API_KEY environment variable.")


# ---------------------------------------------------------------------------
# Hotkey Mapping
# ---------------------------------------------------------------------------

HOTKEY_MAP: dict[str, Key] = {
    "f1": Key.f1,
    "f2": Key.f2,
    "f3": Key.f3,
    "f4": Key.f4,
    "f5": Key.f5,
    "f6": Key.f6,
    "f7": Key.f7,
    "f8": Key.f8,
    "f9": Key.f9,
    "f10": Key.f10,
    "f11": Key.f11,
    "f12": Key.f12,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TranscriptionError(Exception):
    """Base exception for transcription errors."""

    pass


class ApiError(TranscriptionError):
    """API-related errors."""

    pass


class NetworkError(TranscriptionError):
    """Network-related errors."""

    pass


class AudioError(TranscriptionError):
    """Audio recording errors."""

    pass


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------


class AppState(Enum):
    """Application states."""

    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()


# ---------------------------------------------------------------------------
# macOS Notifications
# ---------------------------------------------------------------------------


def notify(title: str, message: str) -> None:
    """Show macOS notification (non-blocking)."""
    safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
    safe_message = message.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{safe_message}" with title "{safe_title}"'
    subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_system_requirements() -> list[str]:
    """Check system requirements and return list of issues found."""
    issues: list[str] = []
    if shutil.which("ffmpeg") is None:
        issues.append("FFmpeg is not installed (required for audio compression).\n  Install with: brew install ffmpeg")
    try:
        devices = sd.query_devices()
        has_input = any(d.get("max_input_channels", 0) > 0 for d in devices) if isinstance(devices, list) else True
        if not has_input:
            issues.append("No audio input device (microphone) found.")
    except Exception:
        issues.append("Could not detect audio devices. Check your audio setup.")
    return issues


# ---------------------------------------------------------------------------
# Console Overlay
# ---------------------------------------------------------------------------


class StatusPrinter:
    """Simple console-based status indicator."""

    def __init__(self, hotkey_name: str, cancel_key_name: str) -> None:
        self._hotkey = hotkey_name
        self._cancel_key = cancel_key_name

    def show_recording(self) -> None:
        """Show recording started."""
        print(f"🎙️ RECORDING... (press {self._hotkey.upper()} to stop, {self._cancel_key.upper()} to discard)")
        notify("Voice2Text", "🎙️ Recording started")

    def show_transcribing(self) -> None:
        """Show transcribing status."""
        print("⏳ Transcribing...")

    def show_cancelled(self) -> None:
        """Show cancelled status."""
        print("❌ Cancelled")

    def hide(self) -> None:
        """Clear status (no-op for console output)."""


# ---------------------------------------------------------------------------
# Audio Recorder
# ---------------------------------------------------------------------------


class Recorder:
    """Records audio from microphone."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._logger = logging.getLogger(__name__)
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._recording = False

    @property
    def recording(self) -> bool:
        """Whether audio is currently being recorded."""
        return self._recording

    def start(self) -> None:
        """Start recording."""
        self._frames = []
        # Re-initialize PortAudio so it picks up device changes
        # (e.g. microphone connected/disconnected/switched).
        sd._terminate()
        sd._initialize()
        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        self._recording = True

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Audio callback - called for each audio block."""
        if status:
            self._logger.warning("Audio stream status: %s", status)
        self._frames.append(indata.copy())

    def stop(self) -> Path | None:
        """Stop recording and return path to audio file."""
        if self._stream is None:
            return None

        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._recording = False

        if not self._frames:
            return None

        audio = np.concatenate(self._frames, axis=0)
        duration = len(audio) / self._config.sample_rate

        if duration < self._config.min_duration_sec:
            print(f"  too short ({duration:.1f}s), skipping")
            return None

        # Save to WAV temporarily
        wav_path = self._config.saved_recording_path
        sf.write(str(wav_path), audio, self._config.sample_rate, subtype="PCM_16")

        # Convert to OGG/OPUS for efficient transmission
        try:
            ogg_path = wav_path.with_suffix(".ogg")
            audio_segment = AudioSegment.from_wav(str(wav_path))
            audio_segment.export(str(ogg_path), format="ogg", codec="libopus", bitrate="24k")

            wav_size = wav_path.stat().st_size / 1024
            ogg_size = ogg_path.stat().st_size / 1024
            ratio = wav_size / ogg_size if ogg_size > 0 else 0

            print(f"  💾 Saved: {wav_path} ({wav_size:.1f} KB)")
            print(f"  🗜️  Compressed: {ogg_path} ({ogg_size:.1f} KB, ~{ratio:.1f}x smaller)")

            # Delete temporary WAV file, keep only OGG
            wav_path.unlink()

            return ogg_path
        except Exception as e:
            print(f"  ⚠️  Compression failed: {e}, using WAV")
            return wav_path


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


class Transcriber:
    """Handles audio transcription via API."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model = config.model_alias
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            }
        )

    def transcribe(self, audio_path: Path) -> str:
        """Send audio to API and return transcription text."""
        audio_bytes = audio_path.read_bytes()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        audio_format = "ogg" if audio_path.suffix.lower() == ".ogg" else "wav"

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._config.transcribe_prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "format": audio_format,
                                "data": audio_base64,
                            },
                        },
                    ],
                }
            ],
        }

        try:
            response = self._session.post(
                f"{self._config.api_base_url}/chat/completions",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout as e:
            raise NetworkError("Request timed out. Please try again.") from e
        except requests.exceptions.ConnectionError as e:
            raise NetworkError("Network error. Check your internet connection.") from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 0

            error_map = {
                401: "Invalid API key. Check your VOICE2TEXT_API_KEY.",
                429: "Rate limit exceeded. Please wait a moment.",
                402: "Insufficient credits. Add funds at https://openrouter.ai/",
            }

            if status_code in error_map:
                raise ApiError(error_map[status_code]) from e
            else:
                raise ApiError(f"API error {status_code}: {e}") from e
        except (KeyError, IndexError) as e:
            raise ApiError(f"Unexpected API response format: {e}") from e


# ---------------------------------------------------------------------------
# Hotkey Handling
# ---------------------------------------------------------------------------


class HotkeyHandler:
    """Handles hotkey detection."""

    def __init__(self, config: Config) -> None:
        self._hotkey = HOTKEY_MAP.get(config.hotkey_name, Key.f8)
        self._cancel_key = HOTKEY_MAP.get(config.cancel_key_name, Key.f9)
        self._insert_enabled = config.hotkey_name == "insert"
        self._cancel_insert_enabled = config.cancel_key_name == "insert"
        self._insert_vk = config.insert_vk

    def is_hotkey(self, key: Key | KeyCode | None) -> bool:
        """Check if the pressed key is the configured hotkey."""
        if key == self._hotkey:
            return True
        return self._insert_enabled and hasattr(key, "vk") and key.vk == self._insert_vk

    def is_cancel_key(self, key: Key | KeyCode | None) -> bool:
        """Check if the pressed key is the cancel hotkey."""
        if key == self._cancel_key:
            return True
        return self._cancel_insert_enabled and hasattr(key, "vk") and key.vk == self._insert_vk


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


class App:
    """Main application class."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._overlay = StatusPrinter(config.hotkey_name, config.cancel_key_name)
        self._recorder = Recorder(config)
        self._transcriber = Transcriber(config)
        self._hotkey_handler = HotkeyHandler(config)
        self._state = AppState.IDLE
        self._cancelled = False
        self._lock = threading.Lock()

    def on_press(self, key: Key | KeyCode | None) -> None:
        """Handle key press events."""
        # Check for cancel key first
        if self._hotkey_handler.is_cancel_key(key):
            self._handle_cancel()
            return

        if not self._hotkey_handler.is_hotkey(key):
            return

        with self._lock:
            if self._state == AppState.IDLE:
                self._start_recording()
            elif self._state == AppState.RECORDING:
                self._stop_recording()
            elif self._state == AppState.TRANSCRIBING:
                # Cancel current transcription and immediately start new recording
                self._cancel_transcription(flash=False)
                self._start_recording()

    def _handle_cancel(self) -> None:
        """Handle cancel key press."""
        with self._lock:
            if self._state == AppState.RECORDING:
                audio_path = self._recorder.stop()
                self._state = AppState.IDLE

                if audio_path and audio_path.exists():
                    try:
                        audio_path.unlink()
                        print("🗑️  Recording canceled")
                        notify("Voice2Text", "🗑️ Recording canceled")
                    except OSError:
                        pass

                self._overlay.hide()

            elif self._state == AppState.TRANSCRIBING:
                self._state = AppState.IDLE
                self._cancel_transcription()  # Uses default flash=True

    def _start_recording(self) -> None:
        """Start audio recording."""
        self._state = AppState.RECORDING
        self._cancelled = False
        self._overlay.show_recording()
        self._recorder.start()

    def _stop_recording(self) -> None:
        """Stop audio recording and start transcription."""
        self._state = AppState.TRANSCRIBING
        notify("Voice2Text", "⏳ Recording stopped, transcribing...")
        audio_path = self._recorder.stop()
        threading.Thread(
            target=self._process_transcription,
            args=(audio_path,),
            daemon=True,
        ).start()

    def _cancel_transcription(self, *, flash: bool = True) -> None:
        """Cancel ongoing transcription.

        Args:
            flash: If True, flash the cancelled status. Set to False when
                   immediately starting a new recording.
        """
        print("  cancelled by user")
        self._cancelled = True
        self._overlay.show_cancelled()
        if flash:
            threading.Thread(target=self._flash_cancel, daemon=True).start()

    def _flash_cancel(self) -> None:
        """Flash cancelled status."""
        time.sleep(0.6)
        self._overlay.hide()

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Check if the error is worth retrying."""
        if isinstance(error, NetworkError):
            return True
        if isinstance(error, ApiError):
            cause = error.__cause__
            if isinstance(cause, requests.exceptions.HTTPError) and cause.response is not None:
                code = cause.response.status_code
                return code in (429, 500, 502, 503, 504)
            return False
        return False

    def _process_transcription(self, audio_path: Path | None) -> None:
        """Process audio transcription."""
        if audio_path is None:
            with self._lock:
                self._state = AppState.IDLE
            self._overlay.hide()
            return

        max_attempts = 2
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                self._overlay.show_transcribing()
                text = self._transcriber.transcribe(audio_path)

                with self._lock:
                    if self._cancelled:
                        print("  result canceled")
                        print(f"💾 Recording saved: {audio_path}")
                        print(f"   To retry later: python3 voice2text.py --retry-file '{audio_path}'")
                        # Do NOT reset _state here — the caller already transitioned
                        # (e.g., to RECORDING if user started a new recording).
                        return
                    self._state = AppState.IDLE

                if text:
                    self._handle_success(text, audio_path)
                else:
                    self._handle_empty(audio_path)
                return

            except Exception as e:
                last_error = e
                if attempt < max_attempts and self._is_retryable(e):
                    with self._lock:
                        if self._cancelled:
                            self._handle_error(e, audio_path)
                            return
                    print(f"  ⚠️  attempt {attempt} failed: {e}")
                    print(f"  🔄 retrying...")
                    continue
                break

        self._handle_error(last_error, audio_path)

    def _handle_success(self, text: str, audio_path: Path) -> None:
        """Handle successful transcription."""
        _present_result(text)
        print(f"💾 Recording saved: {audio_path}")
        self._overlay.hide()

    def _handle_empty(self, audio_path: Path) -> None:
        """Handle empty transcription."""
        print("  empty transcription")
        print(f"💾 Recording saved: {audio_path}")
        self._overlay.hide()

    def _handle_error(self, error: Exception, audio_path: Path) -> None:
        """Handle transcription error."""
        print(f"  ❌ transcription error: {error}")
        print(f"\n💡 Recording saved to: {audio_path}")
        print(f"   To retry, run: python3 voice2text.py --retry-file '{audio_path}'")
        print("   Or: python3 voice2text.py --retry")

        with self._lock:
            self._state = AppState.IDLE
        self._overlay.hide()

    def run(self) -> None:
        """Run the application."""
        print("Voice2Text")
        print()

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        try:
            while listener.is_alive():
                listener.join(timeout=0.1)
        except KeyboardInterrupt:
            print("\nBye.")
        finally:
            listener.stop()


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


def _present_result(text: str) -> None:
    """Display and copy a successful transcription result."""
    print(f"✅ Result: {text}")
    pyperclip.copy(text)
    print("📋 Text copied to clipboard (Cmd+V to paste)")
    notify("Voice2Text", "✅ Transcription ready! Text copied to clipboard")


def get_saved_recording(config: Config) -> Path | None:
    """Get the saved recording if it exists (prefers OGG for retry)."""
    if config.saved_ogg_path.exists():
        return config.saved_ogg_path
    if config.saved_recording_path.exists():
        return config.saved_recording_path
    return None


def retry_transcription(audio_path: Path, config: Config) -> None:
    """Retry transcription for a saved audio file."""
    print(f"\n🔄 Retrying transcription for: {audio_path}")
    print(f"Model: {config.model_alias}")
    print()

    try:
        overlay = StatusPrinter(config.hotkey_name, config.cancel_key_name)
        overlay.show_transcribing()

        transcriber = Transcriber(config)
        text = transcriber.transcribe(audio_path)

        if text:
            _present_result(text)
        else:
            print("  empty transcription")

    except TranscriptionError as e:
        print(f"  ❌ transcription error: {e}")
        print("\n💡 To retry again, run:")
        print(f"   python3 voice2text.py --retry-file '{audio_path}'")


def show_saved_recordings(config: Config) -> None:
    """Display info about saved recordings."""
    has_wav = config.saved_recording_path.exists()
    has_ogg = config.saved_ogg_path.exists()

    if not has_wav and not has_ogg:
        print("\nNo saved recordings found.")
        print(f"Recordings are saved to: {config.saved_recording_path}")
        return

    print("\n📁 Saved recordings:")
    print("=" * 70)

    if has_wav:
        size = config.saved_recording_path.stat().st_size / 1024
        mtime = datetime.fromtimestamp(config.saved_recording_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  WAV: {config.saved_recording_path.name}")
        print(f"       Size: {size:.1f} KB")
        print(f"       Date: {mtime}")

    if has_ogg:
        size = config.saved_ogg_path.stat().st_size / 1024
        mtime = datetime.fromtimestamp(config.saved_ogg_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  OGG: {config.saved_ogg_path.name}")
        if has_wav:
            wav_size = config.saved_recording_path.stat().st_size / 1024
            ratio = wav_size / size if size > 0 else 0
            print(f"       Size: {size:.1f} KB ({ratio:.1f}x smaller than WAV)")
        else:
            print(f"       Size: {size:.1f} KB")
        print(f"       Date: {mtime}")

    print("\n" + "=" * 70)
    print("\nTo retry transcription:")
    print("  python3 voice2text.py --retry")
    print(f"  python3 voice2text.py --retry-file '{config.saved_ogg_path.name}'")
    print()


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        prog="voice2text",
        description="Voice-to-Text: hotkey-activated voice transcription tool",
        epilog=(
            "Environment Variables:\n"
            "  VOICE2TEXT_API_KEY         API key (required)\n"
            "  VOICE2TEXT_HOTKEY=f8       Hotkey (f1-f12, insert)\n"
            "  VOICE2TEXT_CANCEL_KEY=f9   Cancel key\n"
            "  VOICE2TEXT_MODEL           Model to use\n"
            "\n"
            "Hotkeys:\n"
            "  F8 (or configured key)     Start/stop recording\n"
            "  F9 (or cancel key)         Discard recording\n"
            "  Ctrl+C                     Quit"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--retry", action="store_true", help="retry transcription of the latest recording")
    group.add_argument("--retry-file", type=Path, metavar="PATH", help="retry transcription of a specific file")
    group.add_argument("--list-recordings", "-r", "--recordings", action="store_true", help="show saved recordings")
    return parser


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def _validate_api_key(config: Config) -> int | None:
    """Validate API key, returning exit code on failure or None on success."""
    try:
        config.validate()
    except ValueError:
        print("❌ Error: VOICE2TEXT_API_KEY environment variable is not set.")
        print("")
        print("To fix this:")
        print("  1. Get your API key from https://openrouter.ai/keys")
        print("  2. Set the environment variable:")
        print("     export VOICE2TEXT_API_KEY='your-key-here'")
        print("")
        print("Or create a .env file in the project root with:")
        print("  VOICE2TEXT_API_KEY=your-key-here")
        return 1
    return None


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Load configuration
    config = Config.from_env()

    # Handle --list-recordings (no API key needed)
    if args.list_recordings:
        show_saved_recordings(config)
        return 0

    # Handle --retry
    if args.retry:
        recording = get_saved_recording(config)
        if not recording:
            print("❌ No saved recording found.")
            print(f"📁 Recording file: {config.saved_recording_path}")
            return 1
        if (rc := _validate_api_key(config)) is not None:
            return rc
        retry_transcription(recording, config)
        return 0

    # Handle --retry-file
    if args.retry_file:
        if not args.retry_file.exists():
            print(f"❌ File not found: {args.retry_file}")
            return 1
        if (rc := _validate_api_key(config)) is not None:
            return rc
        retry_transcription(args.retry_file, config)
        return 0

    # Validate API key before starting
    if (rc := _validate_api_key(config)) is not None:
        return rc

    # Check system requirements
    issues = check_system_requirements()
    if issues:
        for issue in issues:
            print(f"⚠️  {issue}")
        print()

    # Normal recording mode
    app = App(config)
    app.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
