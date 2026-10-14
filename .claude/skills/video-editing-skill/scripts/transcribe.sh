#!/usr/bin/env bash
set -euo pipefail

# transcribe.sh — Transcribe video/audio to SRT using faster-whisper (CUDA, word timestamps)
#
# Usage: ./transcribe.sh <input_file> [options]
#
# Options:
#   --model <model>      Whisper model: tiny, base, small, medium, large-v3 (default: large-v3)
#   --language <lang>    Language code, e.g. ru, en; "auto" to detect (default: ru)
#   --words <n>          Up to n words per cue from word timestamps; 0 = whole segments (default: 0)
#   --output <path>      Output SRT path (default: <input_basename>.srt)
#
# Examples:
#   ./transcribe.sh video.mp4
#   ./transcribe.sh video.mp4 --words 3
#   ./transcribe.sh video.mp4 --language en --output /tmp/captions.srt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# python with faster-whisper: the project venv, <project>/.claude/skills/video-editing-skill/scripts -> <project>/.venv
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
WHISPER_PY="${WHISPER_PY:-$PROJECT_ROOT/.venv/Scripts/python.exe}"
[[ -x "$WHISPER_PY" ]] || WHISPER_PY="$PROJECT_ROOT/.venv/bin/python"

print_usage() {
    sed -n '3,17p' "$0" | sed 's/^# \?//'
}

if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
fi

INPUT_FILE="$1"
shift

MODEL="large-v3"
LANGUAGE="ru"
WORDS=0
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --words)
            WORDS="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown option '$1'"
            print_usage
            exit 1
            ;;
    esac
done

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

if [[ ! -x "$WHISPER_PY" ]]; then
    echo "Error: python with faster-whisper not found at $WHISPER_PY (set WHISPER_PY)"
    exit 1
fi

INPUT_DIR="$(dirname "$INPUT_FILE")"
INPUT_BASENAME="$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')"
FINAL_OUTPUT="${OUTPUT:-${INPUT_DIR}/${INPUT_BASENAME}.srt}"

echo "Transcribing: $INPUT_FILE"
echo "Model: $MODEL (faster-whisper, cuda)"
echo "Language: $LANGUAGE"

[[ "$LANGUAGE" == "auto" ]] && LANGUAGE=""
"$WHISPER_PY" "$SCRIPT_DIR/transcribe_fw.py" "$INPUT_FILE" "$FINAL_OUTPUT" "$MODEL" "$LANGUAGE" "$WORDS"

LINE_COUNT="$(wc -l < "$FINAL_OUTPUT" | tr -d ' ')"
echo "Transcription complete: $FINAL_OUTPUT ($LINE_COUNT lines)"
echo "$FINAL_OUTPUT"
