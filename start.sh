#!/bin/bash
set -e

# Voice2Text Launcher
# Usage: ./start.sh           - Start recording mode
#        ./start.sh r         - Retry saved recording, then start
#        ./start.sh retry     - Same as above

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is required but not installed"
    exit 1
fi

# Check for virtual environment or dependencies
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python3"
elif [ -d "venv" ]; then
    PYTHON="venv/bin/python3"
else
    PYTHON="python3"
fi

# Check if API key is set
if [ -z "$VOICE2TEXT_API_KEY" ]; then
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi
fi

# Check if retry mode is requested
if [ "$1" = "r" ] || [ "$1" = "retry" ]; then
    if [ -f "recording.ogg" ] || [ -f "recording.wav" ]; then
        echo "🔄 Retrying saved recording..."
        $PYTHON voice2text.py --retry
        echo ""
        echo "✅ Retry complete!"
        echo ""
    else
        echo "❌ No saved recording found."
        echo ""
    fi
else
    # Show hint about retry mode only when starting normally
    if [ -f "recording.ogg" ] || [ -f "recording.wav" ]; then
        echo "💡 Tip: Run './start.sh r' to resend previous recording"
        echo ""
    fi
fi

# Always start normal recording mode after retry (or immediately if no retry)
echo "🎙️  Starting Voice2Text..."
$PYTHON voice2text.py
