"""Test talking head without a real face: a drawn robot (MediaPipe detects it) walking around a 1920x1080 frame,
mouth driven by the voice. Usage: F:/work/ebash-montaj/.venv/Scripts/python make_robot.py voice.wav out.mp4"""
import subprocess
import sys

import cv2
import numpy as np
import soundfile as sf

W, H, FPS = 1920, 1080, 30
wav, out = sys.argv[1], sys.argv[2]
a, sr = sf.read(wav, dtype="float32")
dur = len(a) / sr
n = int(dur * FPS)
env = np.array([np.sqrt(np.mean(a[int(i / FPS * sr):int((i + 1) / FPS * sr)] ** 2)) for i in range(n)])
env = np.clip(env / (np.percentile(env, 95) + 1e-6), 0, 1)

# camera-relative path: stand left, walk right mid-phrase, sway, walk back to centre (key times in s)
keys = [(0, 0.28), (7, 0.30), (10, 0.72), (18, 0.70), (19, 0.66), (20, 0.74), (24, 0.72), (27, 0.45)]
keys = [k for k in keys if k[0] < dur] + [(dur, 0.47)]  # a shorter voice just ends the walk early


def path(t):
    ts, xs = zip(*keys)
    return float(np.interp(t, ts, xs))


def robot(img, cx, cy, s, mouth, blink):
    e = lambda c, ax, col, th=-1: cv2.ellipse(img, (int(cx + c[0] * s), int(cy + c[1] * s)),
                                              (int(ax[0] * s), int(ax[1] * s)), 0, 0, 360, col, th, cv2.LINE_AA)
    cv2.rectangle(img, (int(cx - 20 * s), int(cy - 420 * s)), (int(cx + 20 * s), int(cy - 340 * s)), (150, 150, 155), -1)
    cv2.circle(img, (int(cx), int(cy - 430 * s)), int(26 * s), (40, 40, 220), -1, cv2.LINE_AA)
    e((0, 0), (260, 340), (200, 200, 205))
    e((0, 0), (260, 340), (90, 90, 95), max(2, int(8 * s)))
    for dx in (-110, 110):
        e((dx, -70), (62, max(3, 34 * (1 - blink))), (245, 245, 245))
        if blink < 0.5:
            cv2.circle(img, (int(cx + dx * s), int(cy - 70 * s)), int(24 * s), (40, 120, 230), -1, cv2.LINE_AA)
            cv2.circle(img, (int(cx + dx * s), int(cy - 70 * s)), int(10 * s), (10, 10, 10), -1, cv2.LINE_AA)
        cv2.line(img, (int(cx + (dx - 70) * s), int(cy - 135 * s)), (int(cx + (dx + 60) * s), int(cy - 145 * s)),
                 (70, 70, 75), max(2, int(14 * s)), cv2.LINE_AA)
    nose = np.array([[cx, cy - 40 * s], [cx - 38 * s, cy + 70 * s], [cx + 38 * s, cy + 70 * s]], np.int32)
    cv2.polylines(img, [nose], True, (120, 120, 130), max(2, int(6 * s)), cv2.LINE_AA)
    e((0, 165), (95, 8 + 40 * mouth), (60, 60, 160))


enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                        "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", out], stdin=subprocess.PIPE)
bg = np.zeros((H, W, 3), np.uint8)
bg[:] = (52, 44, 38)
for x in range(0, W, 160):  # a wall grid so camera moves are visible
    cv2.line(bg, (x, 0), (x, H), (62, 54, 48), 2)
for i in range(n):
    t = i / FPS
    img = bg.copy()
    walking = abs(path(t + 0.05) - path(t - 0.05)) > 0.002
    bob = 8 * abs(np.sin(t * 9)) if walking else 3 * np.sin(t * 1.3)
    blink = 1.0 if (t % 4.1) < 0.12 else 0.0
    robot(img, path(t) * W, 560 + bob, 0.95, env[i], blink)
    enc.stdin.write(img.tobytes())
enc.stdin.close()
enc.wait()
print(out, f"{dur:.1f}s")
