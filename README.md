# 🎙️ Voice2Text

Press a key, speak, get text. A simple voice input tool for macOS that works in any application.

**Voice2Text solves this:**

- ✅ **Free tier available** — use free models
- ✅ **Cheap high-quality models** — use affordable models (recommended: Gemini 3 Flash, ~$0.002/min, quality better than Wispr Flow, which costs $7/month)
- ✅ **No subscriptions** — pay only for what you use (if anything)
- ✅ **Your own API key** — works with any OpenAI-compatible API (OpenRouter, OpenAI, Anthropic, etc.)
- ✅ **Global hotkey** — press F8, speak, press F8 again — text is in your clipboard

## How It Works

```
Press F8 → Speak → Press F8 again → Text in clipboard!
```

Recording is automatically compressed to OGG/OPUS (10-20x smaller) and sent to your configured API for transcription. The whole process takes 1-3 seconds.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/anoru/voice2text.git
cd voice2text

# 2. Install dependencies
pip3 install -r requirements.txt
brew install ffmpeg  # Required for audio compression

# 3. Configure API (copy .env.example and edit)
cp .env.example .env
# Edit .env and add your API key

# 4. Run!
./start.sh
```

Press **F8**, say something, press **F8** again — text is copied to clipboard!

## Hotkeys

| Key        | Action                                               |
| ---------- | ---------------------------------------------------- |
| **F8**     | Start/stop recording                                 |
| **F9**     | Cancel recording or transcription (saves API tokens) |
| **Ctrl+C** | Quit application                                     |

Hotkeys can be customized via environment variables in `.env` file.

## Installation

### Requirements

- **macOS** (uses `osascript` for notifications)
- Python 3.10+
- Microphone access
- FFmpeg (for audio compression)
- API key from any OpenAI-compatible provider (OpenRouter, OpenAI, Anthropic, etc.)

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/anoru/voice2text.git
cd voice2text

# 2. Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install FFmpeg (required for audio compression)
brew install ffmpeg

# 5. Configure environment
cp .env.example .env
```

## Configuration

All configuration is done via the `.env` file (created from `.env.example`):

```bash
# Copy example file
cp .env.example .env

# Edit .env with your settings
```

### Using Different API Providers

Voice2Text works with any OpenAI-compatible API endpoint. Edit your `.env` file:

**Example with OpenRouter:**

```bash
VOICE2TEXT_API_KEY=sk-or-v1-xxx
VOICE2TEXT_API_URL=https://openrouter.ai/api/v1
VOICE2TEXT_MODEL=google/gemini-3-flash-preview
```

**OpenAI:**

```bash
VOICE2TEXT_API_KEY=sk-xxx
VOICE2TEXT_API_URL=https://api.openai.com/v1
VOICE2TEXT_MODEL=gpt-4o-mini
```

**Anthropic:**

```bash
VOICE2TEXT_API_KEY=sk-ant-xxx
VOICE2TEXT_API_URL=https://api.anthropic.com/v1
VOICE2TEXT_MODEL=claude-3-haiku
```

**Any other provider** — just set the API key and endpoint URL in `.env`.

### Getting API Key

1. Sign up at [OpenRouter](https://openrouter.ai) (or any other provider)
2. Create an API key in your provider's dashboard
3. Open `.env` file and paste your key:
   ```
   VOICE2TEXT_API_KEY=sk-or-v1-your-key-here
   ```
4. Set the endpoint URL:
   ```
   VOICE2TEXT_API_URL=https://openrouter.ai/api/v1
   ```
5. Set the model (check your provider's documentation for available models):
   ```
   VOICE2TEXT_MODEL=google/gemini-3-flash-preview
   ```

## Notifications

Voice2Text provides native macOS notifications throughout the transcription process:

1. **🎙️ Recording Started** — When you press F8 to begin recording
2. **⏳ Transcribing** — When recording stops and audio is being processed
3. **✅ Transcription Ready** — When text is successfully transcribed and copied to clipboard

Notifications help you track the workflow without watching the terminal. They appear in the top-right corner of your screen and automatically dismiss after a few seconds.

## Features

- 🎙️ **Audio compression** — automatic conversion to OGG/OPUS (10-20x smaller file size)
- 🔄 **Retry functionality** — if transcription fails (API error, network issue), your recording is saved locally. Retry with the same or different model without re-recording
- 📋 **Clipboard integration** — result instantly copied to clipboard, paste anywhere
- 🔔 **macOS notifications** — native notifications when transcription is ready
- 💾 **Local save** — recording saved locally in case of API error

## Technical Details

### Architecture

```
Hotkey (F8) → Record Audio → Save as WAV → Compress to OGG/OPUS
     → Send to API → Transcription → Copy to Clipboard
```

1. Press hotkey to start recording
2. Audio captured at 16kHz mono
3. Saved as temporary WAV
4. Compressed to OGG/OPUS using FFmpeg (10-20x smaller)
5. Sent to API with selected model
6. Transcription returned and copied to clipboard
7. macOS notification shown

### Command Line Options

```bash
# Start recording mode
./start.sh

# Retry last saved recording
./start.sh retry
```

### Retry Feature

**Why it's useful:**

Sometimes transcription fails due to:

- API rate limits
- Network connectivity issues
- Temporary service outages
- Choosing the wrong model

**Your recording is never lost.** When an error occurs, Voice2Text automatically saves your audio file locally. You can retry transcription later without re-recording.

**Example scenario:**

1. You record a 2-minute voice memo
2. You stop recording, but the API returns an error
3. Voice2Text saves `recording.ogg` locally
4. You wait a moment, then run: `./start.sh retry`
5. The transcription completes successfully

Or retry with a different model (edit `.env` first):

```bash
# Edit .env and change VOICE2TEXT_MODEL
./start.sh retry
```

### Create an Alias (Optional)

For quick access, create a shell alias to launch Voice2Text with a single letter:

**For Zsh (default on macOS):**

```bash
# Add to ~/.zshrc
echo "alias v='cd ~/path/to/voice2text && ./start.sh'" >> ~/.zshrc
source ~/.zshrc

# Now just type:
v
```

**For Bash:**

```bash
# Add to ~/.bashrc
echo "alias v='cd ~/path/to/voice2text && ./start.sh'" >> ~/.bashrc
source ~/.bashrc

# Now just type:
v
```

## Troubleshooting

### Microphone not found

**Error:** `No input device found`

**Solution:** Check System Preferences → Security & Privacy → Privacy → Microphone and ensure Terminal has access.

### Accessibility permissions

**Error:** `pynput requires accessibility permissions`

**Solution:**

1. System Preferences → Security & Privacy → Privacy → Accessibility
2. Add Terminal (or your IDE) to the list
3. Restart the application

### FFmpeg not found

**Error:** `Compression failed`

**Solution:** Install FFmpeg:

```bash
brew install ffmpeg
```

### API errors

**Error:** `Invalid API key`

**Solution:** Check your `.env` file has `VOICE2TEXT_API_KEY` set correctly.

## Development

### Using the launcher script

```bash
# Make executable and use
chmod +x start.sh
./start.sh

# Retry mode
./start.sh retry
```

### Linting and formatting

```bash
pip install ruff
ruff check .
ruff format .
```

## License

This project is released into the public domain using the [Unlicense](LICENSE). You can do whatever you want with this code — no attribution required.

## Acknowledgments

- [OpenRouter](https://openrouter.ai) for unified API access to AI models
- [pynput](https://github.com/moses-palmer/pynput) for keyboard control
- [sounddevice](https://python-sounddevice.readthedocs.io/) for audio recording
- [pydub](https://github.com/jiaaro/pydub) for audio compression

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/anoru/voice2text/issues) on GitHub.

---

<p align="center">
  Made for people who prefer speaking to typing
</p>
