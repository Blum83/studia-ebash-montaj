"""Talking-head reels montage: drop repeated takes (keeps the last), cut pauses, burn word-level ASS subtitles,
render 1080x1920 H.264 into out/.

Usage:  .venv\\Scripts\\python montage.py video.mp4 [--music] [--makeup] [--dry-run]
        --music:   video has background music -> split voice/music (Demucs), cut only the voice,
                   lay the original track under it uncut
        --makeup:  light makeup on the face before rendering (makeup.py)
        --dry-run: print what would be cut and stop before rendering
Output: out/<name>_edit.mp4, out/<name>.ass; caches in out/cache/ (delete *.words.json to re-transcribe)
Rules:  CLAUDE.md, section "Мои правила монтажа"
"""
import bisect
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

OUT = Path(__file__).parent / "out"
CACHE = OUT / "cache"
W, H = 1080, 1920
MODEL = "large-v3"
LANG = "ru"
PAUSE = 0.4           # gaps longer than this (s) are cut
SILENCE_DB = -35      # below this level counts as a pause; raise (e.g. -30) for noisy rooms
PAD = 0.15            # silence (s) kept on each side of speech at a cut
XFADE = 0.03          # audio crossfade (s) at each cut, no clicks
TAKE_MATCH = 0.6      # share of a phrase found contiguously in a later phrase => failed take
TAKE_MIN_WORDS = 2
TAKE_LOOKAHEAD = 3    # how many phrases ahead to look for the retake
ZOOM = 0.08           # cuts alternate 100% / 108% framing to hide jump cuts
ZOOM_T = 0.5          # zoom transition (s), centred on the cut
OUTRO = 1.0           # tail kept after the last word (s), faded to black / silence
MAX_WORDS = 3         # subtitle lines of 2-3 words
FONT_SIZE = 110
HIGHLIGHT = "&H00FFFF&"  # ASS BGR: yellow
# typical Whisper output on music/noise/silence, never real speech
HALLUCINATION = re.compile(r"субтитры (сделал|создавал|подготовил)|dimatorzok|продолжение следует", re.I)


def load_cuda_dlls():
    # cuBLAS/cuDNN come from pip nvidia-* wheels; ctranslate2 does not look there on Windows
    for sp in site.getsitepackages():
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


def separate(video):
    """Voice and music stems via Demucs htdemucs on GPU, cached in out/cache."""
    voice, music = CACHE / f"{video.stem}.voice.wav", CACHE / f"{video.stem}.music.wav"
    if voice.exists() and music.exists():
        return voice, music
    import numpy as np
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    model = get_model("htdemucs").cuda().eval()
    sr = model.samplerate
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "2", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = torch.from_numpy(np.frombuffer(pcm, np.float32).reshape(-1, 2).T.copy())
    mean, std = x.mean(), x.std()
    with torch.no_grad():
        stems = apply_model(model, ((x - mean) / std)[None].cuda(), device="cuda", overlap=0.25)[0].cpu() * std + mean
    v = stems[model.sources.index("vocals")]
    for path, y in ((voice, v), (music, stems.sum(0) - v)):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
                        str(path)], input=y.T.contiguous().numpy().tobytes(), check=True)
    return voice, music


def silences(audio, dur):
    # apad: video longer than its audio track -> the tail counts as silence, not as sound
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(audio), "-vn", "-af",
                          f"apad=whole_dur={dur},silencedetect=n={SILENCE_DB}dB:d={PAUSE}", "-f", "null", "-"],
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
    model = WhisperModel(MODEL, device="cuda", compute_type="float16")
    clips = [t for a, b in isl for t in (max(0.0, a - PAD), min(dur, b + PAD))]
    segments, _ = model.transcribe(str(audio), language=LANG, word_timestamps=True,
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
                line = " ".join(f"{{\\c{HIGHLIGHT}}}{x['w']}{{\\c&HFFFFFF&}}" if j == k else x["w"]
                                for j, x in enumerate(grp))
                out.append(f"Dialogue: 0,{ts(f(wd['s']))},{ts(f(end))},Default,,0,0,0,,{line}")
    return "\n".join(out) + "\n"


def probe(video):
    j = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json", "-show_entries",
         "stream=width,height,avg_frame_rate:stream_side_data=rotation:format=duration", str(video)],
        capture_output=True, text=True, check=True).stdout)
    s = j["streams"][0]
    w, h = s["width"], s["height"]
    if any(abs(int(sd.get("rotation", 0))) == 90 for sd in s.get("side_data_list", [])):
        w, h = h, w  # phone video stored rotated; ffmpeg autorotates on decode
    n, d = map(int, s["avg_frame_rate"].split("/"))
    return w, h, s["avg_frame_rate"], n / d, float(j["format"]["duration"])


def zoom_expr(cuts):
    """1 + ZOOM on odd pieces; smoothstep from one level to the next across each cut (zoompan input time `it`)."""
    terms = []
    for k, c in enumerate(cuts):
        u = f"clip((it-{c:.4f})/{ZOOM_T}+0.5,0,1)"
        terms.append(f"{ZOOM if k % 2 == 0 else -ZOOM}*(3*pow({u},2)-2*pow({u},3))")
    return "+".join(["1"] + terms)


def audio_graph(iv, src="0:a"):
    """Equal-power crossfade at every cut. Inner pieces are widened by XFADE/2 per side,
    so the overlaps eat exactly that and audio length still equals the video's."""
    x, n = XFADE / 2, len(iv)
    g = f"[{src}]asplit={n}" + "".join(f"[s{k}]" for k in range(n)) + ";"
    for k, (a, b) in enumerate(iv):
        a = a - x if k > 0 else a
        b = b + x if k < n - 1 else b
        g += f"[s{k}]atrim={a:.5f}:{b:.5f},asetpts=PTS-STARTPTS[a{k}];"
    prev = "a0"
    for k in range(1, n):
        g += f"[{prev}][a{k}]acrossfade=d={XFADE}:c1=qsin:c2=qsin[x{k}];"
        prev = f"x{k}"
    return g + f"[{prev}]anull[voice];"


def render(video, iv, ass_text, out, fps_str, fps, stems=None):
    # half-open [a,b); video shifted half a frame so float error never flips a boundary frame
    sel = lambda sh: "+".join(f"gte(t,{a - sh:.5f})*lt(t,{b - sh:.5f})" for a, b in iv)
    total = sum(b - a for a, b in iv)
    cuts = list(itertools.accumulate(b - a for a, b in iv))[:-1]
    end = f"st={max(0, total - OUTRO):.4f}:d={min(OUTRO, total):.4f}"
    # centre crop to 9:16, then zoompan scales to 1080x1920
    # shortcut: zoompan rounds x/y to whole pixels, slight wobble during the zoom; pre-upscale if it shows
    graph = (f"[0:v]fps={fps_str},select='{sel(0.5 / fps)}',setpts=N/FRAME_RATE/TB,"
             f"crop='trunc(min(iw,ih*9/16)/2)*2':'trunc(min(ih,iw*16/9)/2)*2',"
             f"zoompan=z='{zoom_expr(cuts)}':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s={W}x{H}:fps={fps_str},"
             f"subtitles=subs.ass,fade=t=out:{end},format=yuv420p[v];")
    if stems:  # cut voice; the original music runs on uncut from where the edit starts
        graph += (audio_graph(iv, "1:a") + f"[2:a]atrim={iv[0][0]:.5f}:{iv[0][0] + total:.5f},asetpts=PTS-STARTPTS[m];"
                  f"[voice][m]amix=inputs=2:normalize=0,afade=t=out:{end}[a]")
    else:
        graph += audio_graph(iv) + f"[voice]afade=t=out:{end}[a]"
    inputs = ["-i", str(video.resolve())] + [x for p in stems or [] for x in ("-i", str(p.resolve()))]
    with tempfile.TemporaryDirectory() as tmp:
        # relative subtitle name in cwd=tmp: avoids escaping Windows paths inside the filtergraph
        Path(tmp, "subs.ass").write_text(ass_text, encoding="utf-8")
        Path(tmp, "graph.txt").write_text(graph, encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats", *inputs,
                        "-/filter_complex", "graph.txt", "-map", "[v]", "-map", "[a]",
                        "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart", str(out.resolve())], cwd=tmp, check=True)


def main(video, music=False, dry_run=False, makeup=False):
    video = Path(video)
    CACHE.mkdir(parents=True, exist_ok=True)
    w, h, fps_str, fps, dur = probe(video)
    stems = separate(video) if music else None
    speech = stems[0] if stems else video  # with music, pauses and words come from the clean voice
    print(f"{video.name}: {dur:.1f}s, {w}x{h}")
    isl = islands(silences(speech, dur), dur)
    words = transcribe(speech, isl, dur, CACHE / f"{video.stem}{'.voice' if stems else ''}.words.json")
    phs = [p for p in phrases(words) if not HALLUCINATION.search(text(p))]
    kept, cut = drop_retakes(phs)
    real = [w for p in phs for w in p]
    # sound with no real words (laugh, "мм", tone) is kept like speech: only pauses and retakes are cut
    spans = [(p[0]["s"], p[-1]["e"]) for p in kept] + [
        (a, b) for a, b in isl if not any(x["s"] < b + PAD and x["e"] > a - PAD for x in real)]
    iv = keep_intervals(spans, fps, dur)
    if not iv:
        sys.exit("no speech or sound found — nothing to keep")
    total = sum(b - a for a, b in iv)
    takes = sum(p[-1]["e"] - p[0]["s"] for p in cut)

    print(f"\nPLAN: {dur:.1f}s -> {total:.1f}s (cut {dur - total:.1f}s)")
    print(f"  retakes: {len(cut)} ({takes:.1f}s)")
    for p in cut:
        print(f"    {p[0]['s']:7.2f}-{p[-1]['e']:7.2f}  {text(p)}")
    print(f"  pauses > {PAUSE}s and edges: {len(iv) - 1} joins, {dur - total - takes:.1f}s")
    if dry_run:
        return

    ass_text = make_ass(kept, remapper(iv))
    (OUT / f"{video.stem}.ass").write_text(ass_text, encoding="utf-8")
    out = OUT / f"{video.stem}_edit.mp4"
    if makeup:
        from makeup import apply
        video = apply(video)  # same timeline and audio, only the face changes
    render(video, iv, ass_text, out, fps_str, fps, stems)
    print(f"done: {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    flags = {"--music", "--dry-run", "--makeup"}
    args = [a for a in sys.argv[1:] if a not in flags]
    if len(args) != 1 or not Path(args[0]).is_file():
        sys.exit(__doc__)
    main(args[0], "--music" in sys.argv, "--dry-run" in sys.argv, "--makeup" in sys.argv)
