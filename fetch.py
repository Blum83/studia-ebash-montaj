"""Моменты из записи эфира: громкие всплески (крики, реакции на гол) → куски видео + список CLIPS для rl_short.py.

.venv\\Scripts\\python fetch.py <url> [--top 10] [--pre 20] [--post 8] [--no-video]

1. yt-dlp качает только звук эфира (4 ч ≈ 200 МБ), кэш out/streams/<id>.*
2. Громкость по секундам минус фон за минуту вокруг → топ-N всплесков, не ближе друг к другу, чем длина куска.
3. Печатает таймкоды со ссылками (посмотреть до скачивания), качает только эти куски видео
   в out/streams/<id>/<старт>.mp4 и печатает готовый CLIPS.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np

STREAMS = Path(__file__).parent / "out" / "streams"
SR = 8000          # для громкости по секундам хватает
SMOOTH = 3         # с, сглаживание громкости
BASELINE = 61      # с, окно фона (медиана): всплеск считается относительно своей минуты эфира
VIDEO_FMT = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"


def ydl(**opts):
    from yt_dlp import YoutubeDL
    return YoutubeDL({"quiet": True, "no_warnings": True, "noprogress": True} | opts)


def fetch_audio(url):
    STREAMS.mkdir(parents=True, exist_ok=True)
    with ydl(format="bestaudio", outtmpl=str(STREAMS / "%(id)s.%(ext)s")) as y:
        info = y.extract_info(url)
    return info, Path(info["requested_downloads"][0]["filepath"])


def loudness(audio):
    """dB по секундам. shortcut: весь звук в память (4 ч ≈ 230 МБ int16), для эфиров длиннее ~10 ч — читать кусками."""
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(audio), "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    x = x[: len(x) // SR * SR].reshape(-1, SR)
    return 10 * np.log10((x ** 2).mean(1) + 1e-10)


def peaks(db, top, pre, post):
    """Секунды всплесков, сильнейшие первыми; куски [t-pre, t+post] не пересекаются."""
    db = np.convolve(db, np.ones(SMOOTH) / SMOOTH, "same")
    h = BASELINE // 2
    base = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(db, h, mode="edge"), BASELINE), 1)
    score = db - base
    picked = []
    for t in np.argsort(-score):
        if len(picked) == top or score[t] <= 0:
            break
        if all(abs(t - p) >= pre + post for p in picked):
            picked.append(int(t))
    return picked


def hms(t):
    return f"{int(t) // 3600}:{int(t) % 3600 // 60:02}:{int(t) % 60:02}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--pre", type=int, default=20, help="с до всплеска (сам момент обычно раньше крика)")
    ap.add_argument("--post", type=int, default=8, help="с после всплеска")
    ap.add_argument("--no-video", action="store_true", help="только таймкоды, без скачивания кусков")
    a = ap.parse_args()

    info, audio = fetch_audio(a.url)
    print(f"{info['title']} — {hms(info['duration'])}")
    ts = peaks(loudness(audio), a.top, a.pre, a.post)
    spans = [(max(0, t - a.pre), min(info["duration"], t + a.post)) for t in ts]
    for i, (t, (s, _)) in enumerate(zip(ts, spans), 1):
        print(f"{i:2}. {hms(t)}  https://youtu.be/{info['id']}?t={s}")
    if a.no_video:
        return

    from yt_dlp.utils import download_range_func
    folder = STREAMS / info["id"]
    with ydl(format=VIDEO_FMT, merge_output_format="mp4", force_keyframes_at_cuts=True,
             download_ranges=download_range_func(None, spans),
             outtmpl=str(folder / "%(section_start)d.%(ext)s")) as y:
        y.download([a.url])
    print("\nCLIPS = [  # для rl_short.py, порядок — по силе всплеска")
    for s, e in spans:
        print(f'    (r"{folder / f"{int(s)}.mp4"}", 0.0, {e - s:.1f}),')
    print("]")


if __name__ == "__main__":
    main()
