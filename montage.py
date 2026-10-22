"""Talking-head reels montage: drop repeated takes (keeps the last), cut pauses, burn word-level ASS subtitles,
render 1080x1920 H.264 into out/.

Usage:  .venv\\Scripts\\python montage.py video.mp4 [--music] [--makeup] [--denoise] [--dry-run] [--lang=en]
        --music:   video has background music -> split voice/music (Demucs), cut only the voice,
                   lay the original track under it uncut
        --makeup:  light makeup on the face before rendering (makeup.py)
        --denoise: FFT noise reduction on the voice (noisy room, fan, street)
        --dry-run: print what would be cut and stop before rendering
        --lang=xx: speech language other than Russian (e.g. game characters in English)
Output: out/<name>_edit.mp4, out/<name>.ass, out/<name>.srt, out/<name>.edl (cuts for DaVinci/Premiere);
        caches in out/cache/ (delete *.words.json to re-transcribe)
Rules:  CLAUDE.md, section "My editing rules"
"""
import bisect
import functools
import glob
import itertools
import json
import math
import os
import re
import site
import subprocess
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out"
CACHE = OUT / "cache"
W, H = 1080, 1920
MODEL = "large-v3"
LANG = "ru"
PAUSE = 0.4           # gaps longer than this (s) are cut
SILENCE_BELOW = 25    # dB under the recording's loud level (90th pct) counts as a pause; lower it for noisy rooms
PAD = 0.15            # silence (s) kept on each side of speech at a cut
XFADE = 0.03          # audio crossfade (s) at each cut, no clicks
TAKE_MATCH = 0.6      # share of a phrase found contiguously in a later phrase => failed take
TAKE_MIN_WORDS = 2
TAKE_LOOKAHEAD = 3    # how many phrases ahead to look for the retake
ZOOM = 0.08           # cuts alternate 100% / 108% framing to hide jump cuts
ZOOM_T = 0.5          # zoom transition (s), centred on the cut
DRIFT = 0.03          # slow creep-in (share) over a long piece, so a static shot breathes
DRIFT_MIN = 3.0       # pieces at least this long (s) get the drift
TRACK_HZ = 10        # face samples per second for the camera path
CAM_DEAD = 0.04       # camera holds until the face moves this share of the frame width (no swaying along)
CAM_SIGMA = 0.4       # camera path smoothing (s, gaussian, zero-phase)
OUTRO = 1.0           # tail kept after the last word (s), faded to black / silence
MAX_WORDS = 3         # subtitle lines of 2-3 words
FONT_SIZE = 110
HIGHLIGHT = "&H00FFFF&"  # ASS BGR: yellow
POP = 112             # current word grows 100% -> POP% ...
POP_MS = 80           # ... over this many ms
LUFS = -14            # Reels/Shorts/TikTok normalise to about -14 LUFS anyway
PEAK = 0.84           # limiter ceiling, about -1.5 dBFS
MAX_GAIN = 40         # dB; a stream mix at -46 LUFS needs ~32, an empty take is not pushed further into hiss
DENOISE = "afftdn=nf=-25"
LANG_SURE = 0.8       # stop when Whisper is this sure the speech is not LANG
# fillers Whisper writes when prompted for them; "а"/"ну" are real words too, so they stay
FILLER = re.compile(r"^(э+|э+м+|м+|мм+|хм+|а{2,}м*|ам+)$")
FILLER_PROMPT = "Ээ, ну, мм, это самое... Короче, вот."
# typical Whisper output on music/noise/silence, never real speech
HALLUCINATION = re.compile(r"субтитры (сделал|создавал|подготовил)|dimatorzok|продолжение следует", re.I)


@functools.cache
def has_gpu():
    """NVIDIA card with CUDA: Whisper/Demucs on GPU and NVENC. Without one everything still runs, slower, on CPU."""
    import ctranslate2
    return ctranslate2.get_cuda_device_count() > 0


def venc(q):
    """H.264 encoder args at quality q (lower = better): NVENC on NVIDIA, libx264 anywhere else."""
    if has_gpu():
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(q)]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(q)]


def load_cuda_dlls():
    # cuBLAS/cuDNN come from pip nvidia-* wheels; ctranslate2 does not look there on Windows
    if not hasattr(os, "add_dll_directory"):  # Linux/macOS find their libraries themselves
        return
    for sp in site.getsitepackages():
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


def key(video):
    """Cache name: same file name from another day/folder must not reuse this file's transcript or stems."""
    st = Path(video).stat()
    return f"{Path(video).stem}.{st.st_size:x}{int(st.st_mtime):x}"


def separate(video):
    """Voice and music stems via Demucs htdemucs on GPU, cached in out/cache."""
    voice, music = CACHE / f"{key(video)}.voice.wav", CACHE / f"{key(video)}.music.wav"
    if voice.exists() and music.exists():
        return voice, music
    import numpy as np
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("htdemucs").to(dev).eval()
    sr = model.samplerate
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "2", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = torch.from_numpy(np.frombuffer(pcm, np.float32).reshape(-1, 2).T.copy())
    mean, std = x.mean(), x.std()
    with torch.no_grad():
        stems = apply_model(model, ((x - mean) / std)[None].to(dev), device=dev, overlap=0.25)[0].cpu() * std + mean
    v = stems[model.sources.index("vocals")]
    for path, y in ((voice, v), (music, stems.sum(0) - v)):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
                        str(path)], input=y.T.contiguous().numpy().tobytes(), check=True)
    return voice, music


def silence_db(audio):
    """Pause threshold relative to this recording: a quiet stream mix (-46 dB mean) has every word under a
    fixed -35 dB, a loud phone mic has room noise over it."""
    import numpy as np
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(audio), "-vn", "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, np.float32)
    x = x[:len(x) // 2000 * 2000].reshape(-1, 2000)  # 0.25 s windows
    db = 10 * np.log10((x ** 2).mean(1) + 1e-12)
    return float(np.percentile(db, 90)) - SILENCE_BELOW if len(db) else -35.0


def silences(audio, dur):
    # apad: video longer than its audio track -> the tail counts as silence, not as sound
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(audio), "-vn", "-af",
                          f"apad=whole_dur={dur},silencedetect=n={silence_db(audio):.1f}dB:d={PAUSE}", "-f", "null", "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    return list(zip(starts, ends + [dur]))  # trailing silence has no end line


def islands(sil, dur):
    """Sound between silences longer than PAUSE."""
    out, prev = [], 0.0
    for a, b in sil + [(dur, dur)]:
        if a > prev:
            out.append((prev, a))
        prev = b
    return out


def transcribe(audio, isl, dur, cache):
    """Whisper only over the sound islands: its own word timing drifts across pauses."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    load_cuda_dlls()
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    model = WhisperModel(MODEL, device="cuda" if has_gpu() else "cpu", compute_type="float16" if has_gpu() else "int8")
    lang, prob, _ = model.detect_language(decode_audio(str(audio)), vad_filter=True)
    if lang != LANG and prob >= LANG_SURE:
        # forced LANG on other speech makes Whisper silently translate it (seen on an English video)
        sys.exit(f"speech is '{lang}' ({prob:.0%} sure), not '{LANG}': set LANG in montage.py or check the file")
    clips = [t for a, b in isl for t in (max(0.0, a - PAD), min(dur, b + PAD))]
    segments, _ = model.transcribe(str(audio), language=LANG, word_timestamps=True, initial_prompt=FILLER_PROMPT if LANG == "ru" else None,
                                   clip_timestamps=clips or [0, 0.1], condition_on_previous_text=False)
    words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)}
             for seg in segments for w in seg.words]
    cache.write_text(json.dumps(words, ensure_ascii=False, indent=1), encoding="utf-8")
    return words


def phrases(words):
    """Split on pauses > PAUSE and on sentence ends."""
    out = [[]]
    for prev, w in zip([None] + words, words):
        if prev and (w["s"] - prev["e"] > PAUSE or prev["w"][-1:] in ".!?…"):
            out.append([])
        out[-1].append(w)
    return [p for p in out if p]


def text(p):
    return " ".join(w["w"] for w in p)


def norm(phrase):
    return [t for t in (re.sub(r"\W", "", w["w"].lower()) for w in phrase) if t]


def drop_retakes(phs):
    """-> (kept, cut). A phrase whose words reappear in one of the next phrases was a failed take."""
    keep, cut = [], []
    for i, p in enumerate(phs):
        a = norm(p)
        retake = len(a) >= TAKE_MIN_WORDS and any(
            SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size / len(a) >= TAKE_MATCH
            for b in map(norm, phs[i + 1:i + 1 + TAKE_LOOKAHEAD]))
        (cut if retake else keep).append(p)
    return keep, cut


def keep_intervals(spans, fps, dur):
    """Padded (start, end) spans, merged and snapped to the frame grid so audio and video stay in sync."""
    iv = []
    for s, e in sorted(spans):
        a = math.floor(max(0, s - PAD) * fps) / fps
        b = min(dur, math.ceil((e + PAD) * fps) / fps)
        if iv and a <= iv[-1][1]:
            iv[-1][1] = max(iv[-1][1], b)
        else:
            iv.append([a, b])
    if iv:
        iv[-1][1] = min(dur, math.ceil((iv[-1][1] + OUTRO) * fps) / fps)
    return iv


def is_filler(w):
    return bool(FILLER.match(re.sub(r"\W", "", w["w"].lower())))


def subtract(iv, cuts, fps):
    """Remove (s, e) spans from intervals, on the frame grid; drop slivers under 3 frames."""
    out = [list(x) for x in iv]
    for s, e in cuts:
        s, e = math.floor(s * fps) / fps, math.ceil(e * fps) / fps
        nxt = []
        for a, b in out:
            for x, y in ((a, min(b, s)), (max(a, e), b)) if s < b and e > a else ((a, b),):
                if y - x >= 3 / fps:
                    nxt.append([x, y])
        out = nxt
    return out


def tc(frames, base):
    return f"{frames // (3600 * base):02}:{frames // (60 * base) % 60:02}:{frames // base % 60:02}:{frames % base:02}"


def make_edl(iv, fps, name):
    """CMX 3600 EDL of the kept pieces: import into DaVinci Resolve / Premiere and fine-tune by hand."""
    base, rec = round(fps), 3600 * round(fps)  # record side starts at 01:00:00:00
    out = [f"TITLE: {Path(name).stem}", "FCM: NON-DROP FRAME", ""]
    for k, (a, b) in enumerate(iv, 1):
        fa, fb = round(a * fps), round(b * fps)
        out += [f"{k:03}  AX       AA/V  C        {tc(fa, base)} {tc(fb, base)} {tc(rec, base)} {tc(rec + fb - fa, base)}",
                f"* FROM CLIP NAME: {name}", ""]
        rec += fb - fa
    return "\n".join(out)


def make_srt(phs, f):
    def stamp(t):
        ms = round(t * 1000)
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
    cues = [(f(g[0]["s"]), f(g[-1]["e"]), text(g)) for p in phs for g in lines(p)]
    return "".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{t}\n\n" for i, (a, b, t) in enumerate(cues, 1))


def remapper(iv):
    """Source time -> output time."""
    starts = [a for a, _ in iv]
    offs = list(itertools.accumulate((b - a for a, b in iv), initial=0))

    def f(t):
        k = max(bisect.bisect_right(starts, t) - 1, 0)
        return offs[k] + min(max(t, iv[k][0]), iv[k][1]) - iv[k][0]
    return f


def ts(t):
    cs = round(t * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02}:{cs // 100 % 60:02}.{cs % 100:02}"


def lines(p):
    """Split a phrase into lines of 2-3 words (no lone word unless the phrase is one word)."""
    k = math.ceil(len(p) / MAX_WORDS)
    return [p[i * len(p) // k:(i + 1) * len(p) // k] for i in range(k)]


def make_ass(phs, f):
    out = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0", "",
           "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
           "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
           "MarginL, MarginR, MarginV, Encoding",
           f"Style: Default,Arial,{FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,"
           f"{FONT_SIZE // 12},2,5,{W // 15},{W // 15},0,204", "",
           "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for p in phs:
        for grp in lines(p):
            for k, wd in enumerate(grp):
                end = grp[k + 1]["s"] if k + 1 < len(grp) else wd["e"]
                line = " ".join(f"{{\\c{HIGHLIGHT}\\t(0,{POP_MS},\\fscx{POP}\\fscy{POP})}}{x['w']}{{\\r}}"
                                if j == k else x["w"] for j, x in enumerate(grp))
                out.append(f"Dialogue: 0,{ts(f(wd['s']))},{ts(f(end))},Default,,0,0,0,,{line}")
    return "\n".join(out) + "\n"


def probe(video):
    j = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json", "-show_entries",
         "stream=width,height,avg_frame_rate,duration:stream_side_data=rotation:format=duration", str(video)],
        capture_output=True, text=True, check=True).stdout)
    s = j["streams"][0]
    w, h = s["width"], s["height"]
    if any(abs(int(sd.get("rotation", 0))) == 90 for sd in s.get("side_data_list", [])):
        w, h = h, w  # phone video stored rotated; ffmpeg autorotates on decode
    n, d = map(int, s["avg_frame_rate"].split("/"))
    fmt = float(j["format"]["duration"])
    # phone/Telegram video can end before its audio: never cut past the last frame
    return w, h, s["avg_frame_rate"], n / d, min(float(s.get("duration", fmt)), fmt)


def faces(video, iv, w, h):
    """Face centre (0..1, 0..1) sampled TRACK_HZ times a second over each kept piece: per piece a list of
    (t, fx, fy), fx/fy None where no face was found."""
    import cv2
    from makeup import landmarker
    mp, lm = landmarker()
    cap, out, t_ms = cv2.VideoCapture(str(video)), [], 0
    for a, b in iv:
        # one seek per piece, then frames in order: a seek per sample decodes from the keyframe every time
        cap.set(cv2.CAP_PROP_POS_MSEC, a * 1000)
        samples, track = list(np.arange(a, b, 1 / TRACK_HZ)), []
        while len(track) < len(samples) and cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # this frame's time
            j = len(track)
            while j < len(samples) and samples[j] <= t + 0.5 / TRACK_HZ:
                j += 1
            due = samples[len(track):j]
            if not due:
                continue
            ok, frame = cap.retrieve()
            t_ms = max(t_ms + 1, int(t * 1000))  # detect_for_video needs increasing timestamps
            res = ok and lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), t_ms)
            p = res.face_landmarks[0] if res and res.face_landmarks else None
            fx, fy = (sum(q.x for q in p) / len(p), sum(q.y for q in p) / len(p)) if p else (None, None)
            track += [(s, fx, fy) for s in due]
        track += [(s, None, None) for s in samples[len(track):]]  # video ended early
        out.append(track)
    cap.release()
    lm.close()
    return out


def smooth_path(ts, vals, out_ts, dead, sigma):
    """Camera centre at out_ts from noisy face samples: gaps filled, outliers dropped (median of 5), a dead zone
    (the target only moves once the face is `dead` away, then to the face itself), then a zero-phase gaussian
    (offline, so no lag: the camera eases in a little before the move). None when the piece has no face at all."""
    v = np.array([np.nan if x is None else x for x in vals], float)
    ok = ~np.isnan(v)
    if not ok.any():
        return None
    v = np.interp(np.arange(len(v)), np.flatnonzero(ok), v[ok])
    v = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(v, 2, mode="edge"), 5), axis=1)
    target, c = [], v[0]
    for i, x in enumerate(v):
        settled = i >= TRACK_HZ and np.ptp(v[i - TRACK_HZ:i + 1]) < dead / 4  # the face stayed put for a second
        c = x if abs(x - c) > dead or (settled and abs(x - c) > dead / 8) else c  # finish centring after a walk
        target.append(c)
    s = max(sigma * TRACK_HZ, 1e-6)
    k = np.exp(-0.5 * (np.arange(-int(3 * s) - 1, int(3 * s) + 2) / s) ** 2)
    sm = np.convolve(np.pad(target, len(k) // 2, mode="edge"), k / k.sum(), mode="valid")
    return np.interp(out_ts, ts, sm)


def steps(var, starts, vals):
    """Piecewise-constant ffmpeg expression: vals[k] from starts[k] on."""
    return "+".join([f"{vals[0]:.4f}"] + [f"{vals[k] - vals[k - 1]:.4f}*gte({var},{starts[k]:.4f})"
                                          for k in range(1, len(vals)) if vals[k] != vals[k - 1]])


def framing(tracks, iv, w, h, fps):
    """9:16 crop following the face frame by frame (smooth_path; free jump at cuts, where it is hidden):
    per piece arrays of crop x/y for its output frames, and the face position inside the crop (piece median)
    for the zoom centre. A piece without a face keeps the previous camera (centre at the start)."""
    cw, ch = min(w, h * 9 / 16) // 2 * 2, min(h, w * 16 / 9) // 2 * 2
    xs, ys, px, py, prev = [], [], [], [], (0.5, 0.45)
    for (a, b), tr in zip(iv, tracks):
        ts = [t for t, _, _ in tr] or [a]
        # exactly the frames render()'s select keeps: grid frames k/fps in [a, b) shifted half a frame
        ft = np.arange(math.ceil(a * fps - 0.5), math.ceil(b * fps - 0.5)) / fps
        cx = smooth_path(ts, [x for _, x, _ in tr] or [None], ft, CAM_DEAD, CAM_SIGMA)
        cy = smooth_path(ts, [y for _, _, y in tr] or [None], ft, CAM_DEAD * w / h, CAM_SIGMA)
        cx = np.full(len(ft), prev[0]) if cx is None else cx
        cy = np.full(len(ft), prev[1]) if cy is None else cy
        prev = (cx[-1], cy[-1]) if len(ft) else prev
        x, y = np.clip(cx * w - cw / 2, 0, w - cw), np.clip(cy * h - ch / 2, 0, h - ch)
        xs.append(x)
        ys.append(y)
        seen = [(fx, fy, np.interp(t, ft, x), np.interp(t, ft, y)) for t, fx, fy in tr if fx is not None] if len(ft) else []
        px.append(float(np.median([(fx * w - xx) / cw for fx, _, xx, _ in seen])) if seen else 0.5)
        py.append(float(np.median([(fy * h - yy) / ch for _, fy, _, yy in seen])) if seen else 0.45)
    return cw, ch, xs, ys, px, py


def camera_cmds(xs, ys, fps):
    """sendcmd script: the crop@cam position for every output frame where it changes (half a frame early,
    so float timestamps never miss the frame)."""
    lines, last = [], None
    for n, (x, y) in enumerate(zip(np.concatenate(xs), np.concatenate(ys))):
        cur = (round(x), round(y))
        if cur != last:
            lines.append(f"{max(0.0, (n - 0.5) / fps):.4f} crop@cam x {cur[0]}, crop@cam y {cur[1]};")
            last = cur
    return "\n".join(lines) + "\n"


def zoom_expr(cuts, total, fps):
    """1 + ZOOM on odd pieces, smootherstep (no jolt at either end) from one level to the next across each cut;
    pieces longer than DRIFT_MIN also creep in by DRIFT over their length and drop back at the cut, where the
    jump is hidden. zoompan input time `it`."""
    terms = []
    for k, c in enumerate(cuts):
        u = f"clip((it-{c:.4f})/{ZOOM_T}+0.5,0,1)"
        terms.append(f"{ZOOM if k % 2 == 0 else -ZOOM}*(6*pow({u},5)-15*pow({u},4)+10*pow({u},3))")
    for s, e in zip([0.0] + cuts, cuts + [total]):
        if e - s >= DRIFT_MIN:
            # drops half a frame early: a rounded cut time must never leave the next piece's first frame zoomed
            terms.append(f"{DRIFT}*clip((it-{s:.4f})/{e - s:.4f},0,1)*lt(it,{e - 0.5 / fps:.4f})")
    return "+".join(["1"] + terms)


def audio_graph(iv, src="0:a", pre=""):
    """Equal-power crossfade at every cut. Inner pieces are widened by XFADE/2 per side,
    so the overlaps eat exactly that and audio length still equals the video's."""
    x, n = XFADE / 2, len(iv)
    g = f"[{src}]{pre}asplit={n}" + "".join(f"[s{k}]" for k in range(n)) + ";"
    for k, (a, b) in enumerate(iv):
        a = a - x if k > 0 else a
        b = b + x if k < n - 1 else b
        g += f"[s{k}]atrim={a:.5f}:{b:.5f},asetpts=PTS-STARTPTS[a{k}];"
    prev = "a0"
    for k in range(1, n):
        g += f"[{prev}][a{k}]acrossfade=d={XFADE}:c1=qsin:c2=qsin[x{k}];"
        prev = f"x{k}"
    return g + f"[{prev}]anull[voice];"


def render(video, iv, ass_text, out, fps_str, fps, frame, stems=None, denoise=False):
    # half-open [a,b); video shifted half a frame so float error never flips a boundary frame
    sel = lambda sh: "+".join(f"gte(t,{a - sh:.5f})*lt(t,{b - sh:.5f})" for a, b in iv)
    total = sum(b - a for a, b in iv)
    cuts = list(itertools.accumulate(b - a for a, b in iv))[:-1]
    end = f"st={max(0, total - OUTRO):.4f}:d={min(OUTRO, total):.4f}"
    # 9:16 crop that follows the face every frame (camera_cmds; jumps only at cuts, where it is hidden),
    # zoom centred on the face
    # shortcut: crop and zoompan move in whole pixels, a slow pan steps by ~2 output px; pre-upscale if it shows
    cw, ch, xs, ys, px, py = frame
    starts = [0.0] + [c - 0.5 / fps for c in cuts]  # half a frame early, like the crop: switch on the cut frame
    zx = f"clip(({steps('it', starts, px)})*iw-iw/zoom/2,0,iw-iw/zoom)"
    zy = f"clip(({steps('it', starts, py)})*ih-ih/zoom/2,0,ih-ih/zoom)"
    graph = (f"[0:v]fps={fps_str},select='{sel(0.5 / fps)}',setpts=N/FRAME_RATE/TB,sendcmd=f=cam.txt,"
             f"crop@cam={cw:.0f}:{ch:.0f}:{xs[0][0]:.0f}:{ys[0][0]:.0f}:exact=1,"  # exact: odd x too, 1-px pan steps
             f"zoompan=z='{zoom_expr(cuts, total, fps)}':x='{zx}':y='{zy}':d=1:s={W}x{H}:fps={fps_str},"
             f"subtitles=subs.ass,fade=t=out:{end},format=yuv420p[v];")
    pre = f"{DENOISE}," if denoise else ""
    master = f"afade=t=out:{end}[a]"  # loudness is set afterwards by normalize()
    if stems:  # cut voice; the original music runs on uncut from where the edit starts
        graph += (audio_graph(iv, "1:a", pre) + f"[2:a]atrim={iv[0][0]:.5f}:{iv[0][0] + total:.5f},asetpts=PTS-STARTPTS[m];"
                  f"[voice][m]amix=inputs=2:normalize=0,{master}")
    else:
        graph += audio_graph(iv, "0:a", pre) + f"[voice]{master}"
    inputs = ["-i", str(video.resolve())] + [x for p in stems or [] for x in ("-i", str(p.resolve()))]
    with tempfile.TemporaryDirectory() as tmp:
        # relative subtitle name in cwd=tmp: avoids escaping Windows paths inside the filtergraph
        Path(tmp, "subs.ass").write_text(ass_text, encoding="utf-8")
        Path(tmp, "cam.txt").write_text(camera_cmds(xs, ys, fps), encoding="utf-8")
        Path(tmp, "graph.txt").write_text(graph, encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats", *inputs,
                        "-/filter_complex", "graph.txt", "-map", "[v]", "-map", "[a]",
                        *venc(21), "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart", str(out.resolve())], cwd=tmp, check=True)


def loudness(path):
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "ebur128",
                          "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", log)[-1])


def normalize(out):
    """Gain to LUFS + peak limiter, video copied. loudnorm alone stops short: it will not push peaks over TP."""
    tmp = out.with_suffix(".norm.mp4")
    gain = min(LUFS - loudness(out), MAX_GAIN)
    for _ in range(4):  # the limiter shaves a little loudness off; a second, smaller gain closes the gap
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(out), "-c:v", "copy", "-af",
                        f"volume={gain:.2f}dB,alimiter=limit={PEAK}:attack=5:release=50:level=false",
                        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)], check=True)
        miss = LUFS - loudness(tmp)
        if abs(miss) < 0.3:
            break
        gain = min(gain + miss, MAX_GAIN)
    tmp.replace(out)


def main(video, music=False, dry_run=False, makeup=False, denoise=False):
    video = Path(video)
    CACHE.mkdir(parents=True, exist_ok=True)
    w, h, fps_str, fps, dur = probe(video)
    stems = separate(video) if music else None
    speech = stems[0] if stems else video  # with music, pauses and words come from the clean voice
    print(f"{video.name}: {dur:.1f}s, {w}x{h}")
    isl = islands(silences(speech, dur), dur)
    words = transcribe(speech, isl, dur, CACHE / f"{key(video)}{'.voice' if stems else ''}.words.json")
    words = [{**w, "w": w["w"].lstrip("-") or w["w"]} for w in words]  # Whisper splits "А-а-а" into "А", "-а"
    phs = [p for p in phrases(words) if not HALLUCINATION.search(text(p))]
    fillers = [w for p in phs for w in p if is_filler(w)]
    phs = [q for q in ([w for w in p if not is_filler(w)] for p in phs) if q]
    kept, cut = drop_retakes(phs)
    real = [w for p in phs for w in p] + fillers
    # sound with no real words (laugh, "мм", tone) is kept like speech: only pauses and retakes are cut
    spans = [(p[0]["s"], p[-1]["e"]) for p in kept] + [
        (a, b) for a, b in isl if not any(x["s"] < b + PAD and x["e"] > a - PAD for x in real)]
    iv = subtract(keep_intervals(spans, fps, dur), [(w["s"], w["e"]) for w in fillers], fps)
    if not iv:
        sys.exit("no speech or sound found — nothing to keep")
    total = sum(b - a for a, b in iv)
    takes = sum(p[-1]["e"] - p[0]["s"] for p in cut)

    print(f"\nPLAN: {dur:.1f}s -> {total:.1f}s (cut {dur - total:.1f}s)")
    print(f"  retakes: {len(cut)} ({takes:.1f}s)")
    for p in cut:
        print(f"    {p[0]['s']:7.2f}-{p[-1]['e']:7.2f}  {text(p)}")
    fill = sum(w["e"] - w["s"] for w in fillers)
    print(f"  fillers: {len(fillers)} ({fill:.1f}s) {' '.join(w['w'] for w in fillers)}")
    print(f"  pauses > {PAUSE}s and edges: {len(iv) - 1} joins, {dur - total - takes - fill:.1f}s")
    if dry_run:
        return

    f = remapper(iv)
    ass_text = make_ass(kept, f)
    (OUT / f"{video.stem}.ass").write_text(ass_text, encoding="utf-8")
    (OUT / f"{video.stem}.srt").write_text(make_srt(kept, f), encoding="utf-8")
    (OUT / f"{video.stem}.edl").write_text(make_edl(iv, fps, video.name), encoding="utf-8")
    frame = framing(faces(video, iv, w, h), iv, w, h, fps)
    out = OUT / f"{video.stem}_edit.mp4"
    if makeup:
        from makeup import apply
        video = apply(video)  # same timeline and audio, only the face changes
    render(video, iv, ass_text, out, fps_str, fps, frame, stems, denoise)
    normalize(out)
    print(f"done: {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    flags = {"--music", "--dry-run", "--makeup", "--denoise"}
    for a in sys.argv[1:]:
        if a.startswith("--lang="):
            LANG = a.split("=", 1)[1]
    args = [a for a in sys.argv[1:] if a not in flags and not a.startswith("--lang=")]
    if len(args) != 1 or not Path(args[0]).is_file():
        sys.exit(__doc__)
    main(args[0], *(f in sys.argv for f in ("--music", "--dry-run", "--makeup", "--denoise")))
