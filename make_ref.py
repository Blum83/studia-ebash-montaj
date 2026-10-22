"""Your recording -> a voice sample for demo.py to clone (folder with ref.wav + ref.txt).
Usage: .venv\\Scripts\\python make_ref.py <recording> <lang> <voice_dir> [start_s]
Takes 7-14 s ending on a sentence end, from the first word at or after start_s. Both cuts go to the quietest point
between words: a sample starting on the tail of the previous word makes the clone start every phrase with that
sound ("ый"). 0.15 s of silence in front, loudness evened out. Check ref.txt: a Whisper typo there worsens the clone.
Record 20-40 s in your own words, in the tone the video needs, with a phone voice recorder (Telegram voice
messages are heavily compressed), in a quiet furnished room."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import montage

SR = 24000


def quietest(a, t0, t1):
    """Time of the quietest 10 ms in [t0, t1]."""
    hop = SR // 100
    ts = np.arange(t0, max(t1, t0 + 0.01), 0.01)
    return float(ts[np.argmin([np.sqrt(np.mean(a[int(t * SR):int(t * SR) + hop] ** 2) + 1e-12) for t in ts])])


def main(src, lang, voice, start=0.0):
    montage.load_cuda_dlls()
    from faster_whisper import WhisperModel
    gpu = montage.has_gpu()
    m = WhisperModel(montage.MODEL, device="cuda" if gpu else "cpu", compute_type="float16" if gpu else "int8")
    segs, _ = m.transcribe(src, language=lang, word_timestamps=True)
    words = [w for s in segs for w in s.words]
    i0 = next(i for i, w in enumerate(words) if w.start >= start)
    s0 = words[i0].start
    ends = [i for i, w in enumerate(words) if i >= i0 and w.word.strip()[-1:] in ".!?…" and 7 <= w.end - s0 <= 14]
    i1 = ends[-1] if ends else max(i for i, w in enumerate(words) if i >= i0 and w.end - s0 <= 14)
    a = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                                     capture_output=True, check=True).stdout, np.float32).copy()
    a0 = quietest(a, words[i0 - 1].end if i0 else max(0.0, s0 - 0.3), s0)
    a1 = quietest(a, words[i1].end, words[i1 + 1].start if i1 + 1 < len(words) else words[i1].end + 0.3)
    seg = a[int(a0 * SR):int(a1 * SR)].copy()
    f = int(0.02 * SR)
    seg[:f] *= np.linspace(0, 1, f)
    seg[-f:] *= np.linspace(1, 0, f)
    seg = np.concatenate([np.zeros(int(0.15 * SR), np.float32), seg, np.zeros(int(0.2 * SR), np.float32)])
    voice = Path(voice)
    voice.mkdir(parents=True, exist_ok=True)
    sf.write(voice / "raw.wav", seg, SR)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(voice / "raw.wav"), "-af", "loudnorm=I=-18:TP=-2",
                    "-ar", str(SR), str(voice / "ref.wav")], check=True)
    (voice / "raw.wav").unlink()
    text = "".join(w.word for w in words[i0:i1 + 1]).strip()
    (voice / "ref.txt").write_text(text, encoding="utf-8")
    print(f"{voice}: {a1 - a0:.1f}s | {text}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
