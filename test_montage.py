"""python test_montage.py — checks take/pause/subtitle logic without GPU or ffmpeg."""
import re
from montage import OUTRO, XFADE, audio_graph, drop_retakes, islands, keep_intervals, lines, phrases, remapper


def W(text, start, step=0.3):
    return [{"w": t, "s": start + i * step, "e": start + i * step + 0.25} for i, t in enumerate(text.split())]


words = (W("Всем привет, сегодня расскажу", 1.0)        # failed take
         + W("Всем привет, сегодня расскажу про монтаж.", 3.0)
         + W("Это просто.", 6.0)                         # 6.0..6.55, then 1s pause
         + W("Три шага.", 7.55))
phs, cut = drop_retakes(phrases(words))
assert [w["w"] for w in phs[0]][:2] == ["Всем", "привет,"] and phs[0][0]["s"] == 3.0, phs[0]
assert len(phs) == 3 and len(cut) == 1 and cut[0][0]["s"] == 1.0

iv = keep_intervals([(p[0]["s"], p[-1]["e"]) for p in phs], fps=25, dur=10)
assert [round(a, 2) for a, _ in iv] == [2.84, 5.84, 7.4], iv  # retake + both long pauses cut, frame-snapped
f = remapper(iv)
assert f(iv[0][0]) == 0 and abs(f(7.55) - (iv[0][1] - iv[0][0] + iv[1][1] - iv[1][0] + 7.55 - iv[2][0])) < 1e-9
assert abs(iv[-1][1] - (8.28 + OUTRO)) < 1e-9, iv  # outro tail kept after the last word

# natural speech with no repeats is untouched
assert len(drop_retakes(phrases(W("Раз два три четыре пять шесть", 0)))[0]) == 1
# crossfades eat exactly the widening: audio length == video length
pieces = [float(b) - float(a) for a, b in re.findall(r"atrim=([\d.]+):([\d.]+)", audio_graph(iv))]
assert abs(sum(pieces) - XFADE * (len(iv) - 1) - sum(b - a for a, b in iv)) < 1e-4, pieces

# sound islands = gaps between silences
assert islands([(0, 1), (3, 4)], 6) == [(1, 3), (4, 6)] and islands([], 5) == [(0.0, 5)]
# subtitle lines: 2-3 words, never a lone word left over
assert [len(g) for g in lines(list(range(7)))] == [2, 2, 3]
assert [len(g) for g in lines(list(range(4)))] == [2, 2] and [len(g) for g in lines([0])] == [1]

# fetch: loud spikes over a quiet stream, strongest first, no overlapping clips
import numpy as np
from fetch import peaks
db = np.full(600, -40.0); db[100] = db[105] = db[400] = 0; db[400] = 10
r = peaks(db, 5, 20, 8)
assert r[0] == 400 and abs(r[1] - 102) <= 4, r  # smoothing may shift a spike by a second
assert all(abs(a - b) >= 28 for a in peaks(db, 5, 20, 8) for b in peaks(db, 5, 20, 8) if a != b)
assert peaks(np.full(300, -30.0), 5, 20, 8) == []  # flat audio: nothing stands out

# yt_check: ids from links, ISO durations, region block / Shorts length flags
from yt_check import problems, seconds, vid
assert vid("https://youtube.com/shorts/a9hQ9mw0tbw") == vid("https://youtu.be/a9hQ9mw0tbw?t=3") == "a9hQ9mw0tbw"
assert seconds("PT2H52M53S") == 10373 and seconds("PT45S") == 45 and seconds("P0D") == 0
v = {"status": {"uploadStatus": "processed", "privacyStatus": "public"},
     "contentDetails": {"duration": "PT4M", "regionRestriction": {"blocked": ["DE", "RU"]}},
     "fileDetails": {"videoStreams": [{"widthPixels": 1080, "heightPixels": 1920}]}}
p = problems(v)
assert len(p) == 2 and "2 странах" in p[0] and "Shorts" in p[1], p

# fillers: cut out of the kept pieces on the frame grid, slivers dropped
from montage import camera_cmds, framing, is_filler, make_edl, smooth_path, steps, subtract
assert [is_filler({"w": x}) for x in ("Ээ,", "ммм", "хм", "а", "ну", "аам", "эм")] == [1, 1, 1, 0, 0, 1, 1]
assert subtract([[0, 4]], [(1.0, 1.4)], 25) == [[0, 1.0], [1.4, 4]]
assert subtract([[0, 4]], [(0.02, 0.5)], 25) == [[0.52, 4]]  # 0..0.0 sliver dropped
# EDL: frame-exact source/record timecodes
edl = make_edl([[1.0, 2.0], [3.0, 3.48]], 25, "v.mp4")
assert "00:00:01:00 00:00:02:00 01:00:00:00 01:00:01:00" in edl and "01:00:01:00 01:00:01:12" in edl, edl
# face framing: crop follows the face but stays inside the frame
import numpy as np
still = lambda a, fx: [(a + k / 10, fx, 0.5) for k in range(10)]
cw, ch, xs, ys, px, py = framing([still(0, 0.9), still(1, 0.5)], [[0, 1], [1, 2]], 1920, 1080, 25)
assert (cw, ch) == (606, 1080) and len(xs[0]) == 25 and np.allclose(xs[0], 1920 - 606) and np.allclose(xs[1], 1920 / 2 - 303)
assert abs(px[1] - 0.5) < 1e-9 and px[0] > 0.5  # pinned to the edge: the face sits right of the crop centre
# a piece with no face keeps the previous camera
cw, ch, xs, ys, px, py = framing([still(0, 0.3), [(1.0, None, None)]], [[0, 1], [1, 2]], 1920, 1080, 25)
assert np.allclose(xs[1], xs[0][-1])
# camera path: sway inside the dead zone -> camera still; a walk -> smooth, monotonic, ends on the face
ts = np.arange(0, 6, 0.1)
sway = smooth_path(ts, 0.5 + 0.015 * np.sin(ts * 3), ts, 0.04, 0.4)
assert np.ptp(sway) < 1e-9
walk = smooth_path(ts, np.interp(ts, [0, 2, 4, 6], [0.3, 0.3, 0.7, 0.7]), ts, 0.04, 0.4)
assert np.all(np.diff(walk) >= -1e-9) and abs(walk[-1] - 0.7) < 0.006 and abs(walk[0] - 0.3) < 0.01
assert np.abs(np.diff(walk)).max() < 0.04  # no jumps between samples
assert smooth_path([0, 0.1], [None, None], [0], 0.04, 0.4) is None
# crop commands only where the rounded position changes
cmds = camera_cmds([np.array([10.2, 10.4, 12.0])], [np.array([0.0, 0.0, 0.0])], 25).splitlines()
assert cmds == ["0.0000 crop@cam x 10, crop@cam y 0;", "0.0600 crop@cam x 12, crop@cam y 0;"], cmds
assert steps("t", [0, 1, 2], [5, 5, 7]) == "5.0000+2.0000*gte(t,2.0000)"
# zoom: the ffmpeg expression evaluated as Python (same syntax): levels alternate, smooth across a cut,
# long pieces creep in by DRIFT and drop back at the cut
from montage import DRIFT, ZOOM, ZOOM_T, zoom_expr
# eval: only our own generated arithmetic, test-only
z = lambda e, t: eval(e, {"clip": lambda x, a, b: min(max(x, a), b), "pow": pow, "lt": lambda a, b: float(a < b), "it": t})
e = zoom_expr([5.0, 6.0], 10.0, 25)  # pieces 0-5 (long), 5-6 (short), 6-10 (long)
assert abs(z(e, 0) - 1) < 1e-9 and abs(z(e, 5.5) - (1 + ZOOM)) < 1e-9 and abs(z(e, 7.0) - 1 - DRIFT / 4) < 1e-6
assert abs(z(e, 5 - ZOOM_T / 2 - 0.001) - (1 + DRIFT * (5 - ZOOM_T / 2 - 0.001) / 5)) < 1e-6  # drift only, before the turn
assert abs(z(e, 5.0) - (1 + ZOOM / 2)) < 1e-9  # the cut frame itself carries no drift any more
zs = [z(e, 4.0 + k / 100) for k in range(151)]  # through the first cut: rises, then the drift drops back at the cut
assert max(abs(b - a) for a, b in zip(zs, zs[1:])) < DRIFT + 0.01 and abs(zs[-1] - (1 + ZOOM)) < 1e-6
# cache key: same file name with other content -> other cache entry
import os, tempfile
from montage import key
d1, d2 = tempfile.mkdtemp(), tempfile.mkdtemp()
open(os.path.join(d1, "IMG_0001.mp4"), "wb").write(b"a"); open(os.path.join(d2, "IMG_0001.mp4"), "wb").write(b"bb")
assert key(os.path.join(d1, "IMG_0001.mp4")) != key(os.path.join(d2, "IMG_0001.mp4"))

# rl_short: every cut lands on a beat, clip ends (the goal) untouched, starts never below 0
from rl_short import on_beat
grid = np.arange(0, 30, 0.4)
cl = on_beat([("a", 2.0, 5.1), ("b", 0.0, 3.0), ("c", 0.1, 4.0)], grid)
cuts = np.cumsum([e - s for _, s, e in cl])
assert all(min(abs(grid - c)) < 1e-9 for c in cuts), cuts
assert [e for _, _, e in cl] == [5.1, 3.0, 4.0] and all(s >= 0 for _, s, _ in cl), cl
print("ok")

# makeup: tint moves colour only inside the mask and keeps lightness roughly
from makeup import tint
img = np.full((20, 20, 3), 128, np.uint8); mask = np.zeros((20, 20), np.float32); mask[5:15, 5:15] = 1
before = img.copy(); tint(img, mask, (200, 30, 60), 0.5)
assert (img[0, 0] == before[0, 0]).all() and img[10, 10, 0] > img[10, 10, 1], img[10, 10]
tint(img, np.zeros((20, 20), np.float32), (0, 0, 0), 1.0)  # empty mask: no-op, no crash
print("ok makeup")
