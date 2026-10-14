"""Rocket League goals short: trimmed replays back to back, 16:9 gameplay over a blurred 9:16 fill,
song's chorus as music, game sound under it (GAME_VOL, 0 = muted). Output: out/rl_short.mp4 (1080x1920, H.264, 60 fps).

Usage: .venv\\Scripts\\python rl_short.py
"""
import glob
import itertools
import subprocess
from pathlib import Path

import numpy as np

from montage import CACHE, normalize

SRC = Path(r"F:\Rocket League")
# (file, start, end) — replays start with the previous goal's "SCORED!" banner, so their first ~2 s are cut
CLIPS = [
    ("Rocket League_2024.04.09-22.28.mp4", 2.0, 9.27),
    ("Rocket League_2024.04.30-00.10.mp4", 0.0, 6.2),
    ("Rocket League_2024.05.03-22.24.mp4", 11.5, 16.9),
    ("Rocket League_2024.05.08-09.27.mp4", 2.0, 9.15),
    ("Rocket League_2024.08.13-16.57.mp4", 0.0, 7.33),
    ("Rocket League_2024.08.28-18.58.mp4", 2.5, 10.17),
    ("2024.11.04-23.18.mp4", 0.0, 5.5),
    ("2024.11.04-23.30.mp4", 2.0, 9.17),
    ("Rocket League_2024.11.05-10.57.mp4", 0.0, 6.5),
]
# glob: the "É" in the path is stored decomposed on disk and does not survive being typed
MUSIC = glob.glob(r"F:\Daunloads\*\Spotify\Playlists\Blumination*\42.*APT.mp3")[0]
# APT. choruses ("Don't you want me..." + "아파트" hook), 16 bars each, found by lyrics + chroma match;
# bar-aligned so the joins stay on the beat. Chained, repeated until the video ends.
CHORUSES = [(32.276, 58.05), (70.95, 96.72)]
MUSIC_VOL = 1.0
GAME_VOL = 0.0         # game sound is kept by default (0.45); muted here on request
FADE = 0.03           # fade at each music join, no clicks
OUTRO = 1.0           # fade to black / silence at the end
W, H, FPS = 1080, 1920, 60
OUT = Path(__file__).parent / "out" / "rl_short.mp4"


def song_beats():
    cache = CACHE / (Path(MUSIC).stem + ".beats.npy")
    if not cache.exists():
        import librosa
        y, sr = librosa.load(MUSIC, sr=22050, mono=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cache, librosa.beat.beat_track(y=y, sr=sr, units="time")[1])
    return np.load(cache)


def music_parts(total):
    parts, left = [], total
    for a, b in itertools.cycle(CHORUSES):
        parts.append((a, min(b, a + left)))
        left -= parts[-1][1] - a
        if left <= 0:
            return parts


def out_beats(parts, beats):
    """Song beats mapped onto the output timeline of the chained chorus parts."""
    out, off = [], 0.0
    for a, b in parts:
        out += [off + t - a for t in beats if a <= t < b]
        off += b - a
    return np.array(out)


def on_beat(clips, grid, min_len=1.0, fps=FPS):
    """Shift each clip's start (never its end: the goal) so every cut lands on the nearest beat,
    rounded to a whole output frame so the video does not drift from the music cut by cut."""
    out, cum = [], 0.0
    for f, s, e in clips:
        ok = [t for t in grid if cum + min_len <= t <= cum + e]  # keeps start >= 0 and the clip >= min_len
        t = min(ok, key=lambda t: abs(t - (cum + e - s))) if ok else cum + e - s
        t = round(t * fps) / fps
        out.append((f, e - (t - cum), e))
        cum = t
    return out


def main():
    n = len(CLIPS)
    clips = on_beat(CLIPS, out_beats(music_parts(sum(e - s for _, s, e in CLIPS) + 10), song_beats()))
    total = sum(e - s for _, s, e in clips)
    g = []
    for i, (_, s, e) in enumerate(clips):
        # fps first, then pad the last frame and cut to the exact length: VFR sources otherwise lose
        # a frame per clip and the picture drifts ahead of the music
        g.append(f"[{i}:v]fps={FPS},trim=start={s:.4f},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=0.2,"
                 f"trim=duration={e - s:.4f},scale=2560:1440,setsar=1[v{i}];"
                 f"[{i}:a]atrim={s:.4f}:{e},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,"
                 f"afade=t=in:d={FADE},afade=t=out:st={e - s - FADE:.3f}:d={FADE}[a{i}];")
    g.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[cv][ca];")
    g.append(f"[cv]split[bg][fg];"
             f"[bg]scale=-2:{H},crop={W}:{H},gblur=sigma=30,eq=brightness=-0.15[bgb];"
             f"[fg]scale={W}:-2[fgs];"
             f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,fade=t=out:st={total - OUTRO:.3f}:d={OUTRO},format=yuv420p[v];")
    parts = music_parts(total)
    k = len(parts)
    g.append(f"[{n}:a]asplit={k}" + "".join(f"[m{j}]" for j in range(k)) + ";")
    for j, (a, b) in enumerate(parts):
        g.append(f"[m{j}]atrim={a}:{b:.3f},asetpts=PTS-STARTPTS,"
                 f"afade=t=in:d={FADE},afade=t=out:st={b - a - FADE:.3f}:d={FADE}[p{j}];")
    g.append("".join(f"[p{j}]" for j in range(k)) + f"concat=n={k}:v=0:a=1,aresample=48000,volume={MUSIC_VOL}[m];"
             f"[ca]volume={GAME_VOL}[game];[m][game]amix=inputs=2:normalize=0,"
             f"afade=t=out:st={total - OUTRO:.3f}:d={OUTRO}[a]")
    inputs = [x for f, _, _ in CLIPS for x in ("-i", str(SRC / f))] + ["-i", MUSIC]
    OUT.parent.mkdir(exist_ok=True)
    graph = OUT.with_suffix(".graph.txt")
    graph.write_text("".join(g), encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats", *inputs,
                    "-/filter_complex", str(graph), "-map", "[v]", "-map", "[a]",
                    "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(OUT)], check=True)
    graph.unlink()
    normalize(OUT)
    shift = max(abs(a[1] - b[1]) for a, b in zip(CLIPS, clips))
    print(f"done: {OUT} ({total:.1f}s, {n} goals, cuts on the beat, starts moved <= {shift:.2f}s)")


if __name__ == "__main__":
    main()
