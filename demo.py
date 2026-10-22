"""Feature demo video (16:9): slides / screenshots / screen clips with an eased zoom into a region, soft crossfades,
voice-over and word-by-word captions. The week's content lives in a script JSON (see examples/robot_demo.json).

Usage:  .venv\\Scripts\\python demo.py <script.json> [--lang ru|uk|en] [--voice DIR] [--design male] [--speed 1.0]
                                       [--tts omni|eleven]
        --voice:  folder with ref.wav + ref.txt = the voice to clone (make_ref.py makes one from your recording);
                  default out/voices/<design>-<lang>, created once from --design and then reused, so the voice
                  stays the same week to week
        --tts eleven: ElevenLabs instead of local OmniVoice (key in secrets/elevenlabs.key, voice id: env ELEVEN_VOICE)
Output: <script folder>/demo_<lang>.mp4 + .srt; voice cache in <script folder>/cache/
OmniVoice runs in its own venv .venv-tts (Python 3.13): see README, "Demo videos with voice-over".
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

import montage  # captions, Whisper, encoder, loudness

REPO = Path(__file__).parent
OMNI_PY = REPO / ".venv-tts" / "Scripts" / "python.exe"
W, H, FPS, SR = 1920, 1080, 30, 44100
TAIL, XF = 0.6, 0.6      # pause after each segment, crossfade between shots (s)
FADE = 1.0               # fade from / to black at the ends (CLAUDE.md rule 4)
MOVE = 1.2               # zoom into the shot's roi over this long (s)
BG = (11, 11, 11)
KEY = REPO / "secrets" / "elevenlabs.key"


def canvas(img):
    """Pad to 16:9 on BG so any 16:9 roi (incl. the full frame) fits; returns image and its offset."""
    h, w = img.shape[:2]
    cw, ch = max(w, round(h * 16 / 9)), max(h, round(w * 9 / 16))
    out = np.full((ch, cw, 3), BG, np.uint8)
    ox, oy = (cw - w) // 2, (ch - h) // 2
    out[oy:oy + h, ox:ox + w] = img
    return out, ox, oy


def fit(roi, ox, oy, cw, ch):
    """Grow roi (x0, y0, x1, y1) to 16:9 around its centre, keep it inside the canvas; returns [x, y, w, h]."""
    x0, y0, x1, y1 = roi[0] + ox, roi[1] + oy, roi[2] + ox, roi[3] + oy
    w, h = x1 - x0, y1 - y0
    w, h = max(w, h * 16 / 9), max(h, w * 9 / 16)
    w, h = min(w, cw), min(h, ch)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return np.array([min(max(cx - w / 2, 0), cw - w), min(max(cy - h / 2, 0), ch - h), w, h], float)


def ease(t):
    return t * t * (3 - 2 * t)


def render_frame(img, rect):
    x, y, w, h = rect
    m = np.float32([[W / w, 0, -x * W / w], [0, H / h, -y * H / h]])
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_AREA if w > W else cv2.INTER_CUBIC, borderValue=BG)


def shot_frames(s, n, start=None):
    """n frames as (frame, camera rect): start on the whole picture (or on `start`, the previous shot's camera when it
    is the same picture), ease into the roi over MOVE, hold; without a roi the picture drifts 100 -> 104%.
    A video shot plays from `ss` with `crop_top` px cut off (e.g. a browser bar)."""
    cap, top = None, s.get("crop_top", 0)
    if s["src"].suffix.lower() in (".mp4", ".mov", ".mkv", ".webm"):
        cap = cv2.VideoCapture(str(s["src"]))
        cap.set(cv2.CAP_PROP_POS_MSEC, s.get("ss", 0) * 1000)
        first = cap.read()[1][top:]
    else:
        first = cv2.imread(str(s["src"]))[top:]
    img, ox, oy = canvas(first)
    ch, cw = img.shape[:2]
    full = fit((-ox, -oy, cw - ox, ch - oy), ox, oy, cw, ch)
    origin = full if start is None else start
    if s.get("roi"):
        target, move = fit(s["roi"], ox, oy, cw, ch), min(MOVE * FPS, n * 0.4)
    else:
        target, move = full * [1, 1, 0.96, 0.96] + [full[2] * 0.02, full[3] * 0.02, 0, 0], n
    for i in range(n):
        if cap is not None and i:
            ok, fr = cap.read()
            if ok:
                img = canvas(fr[top:])[0]
        rect = origin + (target - origin) * (ease(min(i / move, 1.0)) if move else 1.0)
        yield render_frame(img, rect), rect


def norm(w):
    return "".join(c for c in w.lower().replace("ё", "е") if c.isalnum())


def align(shown, heard):
    """Times for the script's words from Whisper's words on the same audio: matched words take Whisper's times,
    runs in between (Whisper writes "чат-спейс" for "ChatSpace") share the gap between their neighbours by length."""
    ws = [{"w": w, "s": None, "e": None} for w in shown.split()]
    a, b = [norm(w["w"]) for w in ws], [norm(h["w"]) for h in heard]
    for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(m.size):
            ws[m.a + k].update(s=heard[m.b + k]["s"], e=heard[m.b + k]["e"])
    i, end = 0, heard[-1]["e"] if heard else 0.0
    while i < len(ws):
        if ws[i]["s"] is not None:
            i += 1
            continue
        j = i
        while j < len(ws) and ws[j]["s"] is None:
            j += 1
        t0 = ws[i - 1]["e"] if i else (heard[0]["s"] if heard else 0.0)
        t1 = ws[j]["s"] if j < len(ws) else max(end, t0)
        total, c = sum(len(w["w"]) for w in ws[i:j]), 0
        for w in ws[i:j]:
            w["s"] = t0 + (t1 - t0) * c / total
            c += len(w["w"])
            w["e"] = t0 + (t1 - t0) * c / total
        i = j
    return ws


def load_wav(p):
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.float32)


def omni(texts, lang, voice, design, speed, cache):
    """OmniVoice in .venv-tts, one model load for all missing texts; cached by voice sample + speed + text.
    Returns (audio, words) per text, words timed by Whisper and mapped onto the script's spelling."""
    if not OMNI_PY.exists():
        raise SystemExit(f"{OMNI_PY} not found: install OmniVoice into .venv-tts (README, 'Demo videos with voice-over')")
    voice.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if not (voice / "ref.wav").exists():  # design the voice once from the first text, then always clone it
        subprocess.run([str(OMNI_PY), str(REPO / "omni_tts.py"), "-", lang, design, str(voice), texts[0]], check=True)
    key = f"{lang}|{speed}|{hashlib.sha1((voice / 'ref.wav').read_bytes()).hexdigest()}"
    outs = [cache / (hashlib.sha1(f"{key}|{t}".encode()).hexdigest()[:16] + ".wav") for t in texts]
    jobs = [{"text": t, "out": str(o), "speed": speed} for t, o in zip(texts, outs) if not o.exists()]
    if jobs:
        (cache / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
        subprocess.run([str(OMNI_PY), str(REPO / "omni_tts.py"), str(cache / "jobs.json"), lang, design, str(voice)],
                       check=True)
    montage.load_cuda_dlls()
    from faster_whisper import WhisperModel
    gpu = montage.has_gpu()
    m = WhisperModel(montage.MODEL, device="cuda" if gpu else "cpu", compute_type="float16" if gpu else "int8")
    res = []
    for t, o in zip(texts, outs):
        segs, _ = m.transcribe(str(o), language=lang, word_timestamps=True, condition_on_previous_text=False)
        res.append((load_wav(o), align(t, [{"w": w.word.strip(), "s": w.start, "e": w.end} for s in segs for w in s.words])))
    return res


def eleven(texts, lang, say, cache):
    """ElevenLabs multilingual_v2 with per-character timestamps; cached by voice + text (credits are spent once).
    `say`: {"Ctrl+W": "контрол дабл-ю"} — spoken differently, captioned as written."""
    import urllib.request
    voice = os.environ.get("ELEVEN_VOICE", "JBFqnCBsd6RMkjVDRZzb")  # premade "George"
    cache.mkdir(parents=True, exist_ok=True)
    res = []
    for text in texts:
        spoken = text
        for written, sp in say.items():
            spoken = spoken.replace(written, sp)
        c = cache / (hashlib.sha1(f"eleven|{voice}|{lang}|{spoken}".encode()).hexdigest()[:16] + ".json")
        if not c.exists():
            body = {"text": spoken, "model_id": "eleven_multilingual_v2", "language_code": lang,
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "speed": 1.0}}
            req = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps"
                                         "?output_format=mp3_44100_128", data=json.dumps(body).encode(),
                                         headers={"xi-api-key": KEY.read_text().strip(), "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                c.write_bytes(r.read())
        r = json.loads(c.read_text(encoding="utf-8"))
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                             input=base64.b64decode(r["audio_base64"]), capture_output=True, check=True).stdout
        al, heard, cur = r["alignment"], [], None
        for ch, s, e in zip(al["characters"], al["character_start_times_seconds"], al["character_end_times_seconds"]):
            if ch.isspace():
                cur = None
            elif cur is None:
                cur = {"w": ch, "s": s, "e": e}
                heard.append(cur)
            else:
                cur["w"] += ch
                cur["e"] = e
        res.append((np.frombuffer(pcm, np.float32), align(text, heard)))  # back to the written form
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--voice")
    ap.add_argument("--design", default="male")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--tts", default="omni", choices=["omni", "eleven"])
    a = ap.parse_args()
    script = Path(a.script).resolve()
    d = script.parent
    sc = json.loads(script.read_text(encoding="utf-8"))
    segs = sc["segments"]
    for sg in segs:
        for s in sg["shots"]:
            s["src"] = d / s["src"]
    texts = [sg["text"][a.lang] for sg in segs]
    if a.tts == "omni":
        voice = Path(a.voice) if a.voice else REPO / "out" / "voices" / f"{a.design}-{a.lang}"
        takes = omni(texts, a.lang, voice, a.design, a.speed, d / "cache")
    else:
        takes = eleven(texts, a.lang, sc.get("say", {}), d / "cache")

    audio, words, plan, t = [], [], [], 0.0
    for sg, (au, ws) in zip(segs, takes):
        words += [{"w": w["w"], "s": round(t + w["s"], 3), "e": round(t + w["e"], 3)} for w in ws]
        audio += [au, np.zeros(int(TAIL * SR), np.float32)]
        plan.append((sg["shots"], t, t + len(au) / SR + TAIL))
        t += len(au) / SR + TAIL
    audio.append(np.zeros(int(FADE * SR), np.float32))  # tail under the fade to black
    total = t + FADE
    out = d / f"demo_{a.lang}.mp4"
    wav = d / "cache" / f"voice_{a.lang}.wav"
    sf.write(wav, np.concatenate(audio), SR)

    # captions: montage's word-pop ASS, re-styled for 16:9 — smaller, at the bottom, on a dark box
    montage.W, montage.H, montage.FONT_SIZE, montage.MAX_WORDS = W, H, 54, 6
    phs = montage.phrases(words)
    ass = montage.make_ass(phs, lambda x: x)
    ass = ass.replace("100,100,0,0,1,4,2,5,", "100,100,0,0,3,10,0,2,").replace(",0,204", ",56,204")
    ass = ass.replace("&H00000000,&H64000000", "&H80000000,&H80000000")
    (d / "cache" / "demo.ass").write_text(ass, encoding="utf-8")
    out.with_suffix(".srt").write_text(montage.make_srt(phs, lambda x: x), encoding="utf-8")

    # each shot gets its share of its segment; consecutive shots of the same picture pan, others crossfade over XF
    # with both sides moving (the outgoing shot renders XF longer)
    shots = []
    for ss, s0, s1 in plan:
        cuts = [s0 + (s1 - s0) * i / len(ss) for i in range(len(ss) + 1)]
        shots += [(s, round(x * FPS), round(y * FPS)) for s, x, y in zip(ss, cuts, cuts[1:])]
    shots[-1] = (shots[-1][0], shots[-1][1], round(total * FPS))
    ov, n_total = round(XF * FPS), round(total * FPS)
    silent = d / "cache" / "video.mp4"
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", *montage.venc(18), "-pix_fmt", "yuv420p", str(silent)],
                           stdin=subprocess.PIPE)
    tail, cam, done = [], None, 0
    for k, (s, f0, f1) in enumerate(shots):
        nxt = shots[k + 1][0] if k + 1 < len(shots) else None
        pan_in = k and s["src"] == shots[k - 1][0]["src"]
        pan_out = nxt is not None and nxt["src"] == s["src"]
        new_tail = []
        for i, (fr, rect) in enumerate(shot_frames(s, f1 - f0 + (0 if pan_out or nxt is None else ov),
                                                   cam if pan_in else None)):
            if i < len(tail):  # crossfade with the outgoing shot's extra frames, eased
                x = ease((i + 1) / (len(tail) + 1))
                fr = cv2.addWeighted(tail[i], 1 - x, fr, x, 0)
            if i >= f1 - f0:
                new_tail.append(fr)
                continue
            cam = rect
            g = min(1.0, (done + 1) / (FADE * FPS), (n_total - done) / (FADE * FPS))
            enc.stdin.write((fr * g).astype(np.uint8).tobytes() if g < 1 else fr.tobytes())
            done += 1
        tail = new_tail
    enc.stdin.close()
    enc.wait()

    # relative subtitle name in cwd: avoids escaping a Windows path inside the filter
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent.name, "-i", wav.name, "-vf", "ass=demo.ass",
                    *montage.venc(20), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", str(out)], cwd=d / "cache", check=True)
    montage.normalize(out)
    print(f"done: {out} ({total:.1f}s)")


if __name__ == "__main__":
    main()
