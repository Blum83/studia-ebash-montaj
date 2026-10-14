"""faster-whisper -> SRT with word-level timing (replaces the openai-whisper CLI).

Usage: python transcribe_fw.py <input> <output.srt> [model] [language] [words_per_cue]
words_per_cue=0 -> one cue per Whisper segment; N -> cues of up to N words cut on word timestamps.
"""
import glob
import os
import site
import sys

src, out = sys.argv[1], sys.argv[2]
model_name = sys.argv[3] if len(sys.argv) > 3 else "large-v3"
lang = (sys.argv[4] if len(sys.argv) > 4 else "ru") or None
per_cue = int(sys.argv[5]) if len(sys.argv) > 5 else 0

# cuBLAS/cuDNN come from pip nvidia-* wheels; ctranslate2 does not look there on Windows
for sp in site.getsitepackages():
    for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel  # noqa: E402

model = WhisperModel(model_name, device="cuda", compute_type="float16")
segments, info = model.transcribe(src, language=lang, word_timestamps=True, vad_filter=True)

cues = []
for seg in segments:
    words = [w for w in seg.words if w.word.strip()]
    if per_cue and words:
        for i in range(0, len(words), per_cue):
            g = words[i:i + per_cue]
            cues.append((g[0].start, g[-1].end, " ".join(w.word.strip() for w in g)))
    else:
        cues.append((seg.start, seg.end, seg.text.strip()))


def stamp(t):
    ms = round(t * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


with open(out, "w", encoding="utf-8") as f:
    f.write("".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{t}\n\n" for i, (a, b, t) in enumerate(cues, 1)))
print(f"{len(cues)} cues, language={info.language}")
