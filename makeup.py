"""Лёгкий макияж на видео: помада, румяна, сглаживание кожи. MediaPipe Face Landmarker (478 точек) + OpenCV.

Отдельно:  .venv\\Scripts\\python makeup.py video.mp4      -> out/cache/<имя>.makeup.mp4
В монтаже: montage.py video.mp4 --makeup                 (кэш тот же; удалить его, если поменял константы)
Цвет меняется в LAB только по каналам a/b — текстура и блики губ/кожи остаются свои.
"""
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from montage import CACHE, key, probe, venc

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL = CACHE / "face_landmarker.task"
DETECT_W = 640              # точки ищем на уменьшенном кадре — быстрее, точность та же (координаты нормированы)

LIP_RGB = (176, 48, 72)     # ягодный
LIP_ALPHA = 0.45            # 0 — нет, 1 — цвет губ целиком заменён
LIP_DARKEN = 0.08           # чуть темнее, как от помады
BLUSH_RGB = (230, 110, 120)
BLUSH_ALPHA = 0.22
BLUSH_R = 0.11              # радиус румян в долях ширины лица
SKIN = 0.45                 # сила сглаживания кожи (bilateral), 0 — выкл.
SMOOTH = 0.55               # вес нового кадра в сглаживании точек: меньше — без дрожи, но отстаёт на резких движениях

LIPS_OUT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_IN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
        149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EYE_L = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_R = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_R = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
CHEEKS = (50, 280)
FACE_W = (234, 454)


def model():
    if not MODEL.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    return MODEL


def landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model())),
                                        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    return mp, vision.FaceLandmarker.create_from_options(opts)


def tint(img, mask, rgb, alpha, darken=0.0):
    """Цвет rgb в зоне mask (0..1): a/b в LAB тянутся к цвету, яркость своя. Работает в рамке маски."""
    ys, xs = np.nonzero(mask > 0.01)
    if not len(ys):
        return
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    roi, m = img[y0:y1, x0:x1], mask[y0:y1, x0:x1, None] * alpha
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    target = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    lab[..., 1:] += (target[1:] - lab[..., 1:]) * m
    lab[..., :1] *= 1 - darken * m / max(alpha, 1e-6)
    img[y0:y1, x0:x1] = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def soft(shape, polys, holes=(), blur=0):
    m = np.zeros(shape[:2], np.float32)
    cv2.fillPoly(m, [p.astype(np.int32) for p in polys], 1.0)
    for p in holes:
        cv2.fillPoly(m, [p.astype(np.int32)], 0.0)
    if blur:
        k = blur | 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return m


def paint(img, pts):
    """Макияж на кадре img (RGB, меняется на месте) по точкам pts (478×2, пиксели)."""
    fw = np.linalg.norm(pts[FACE_W[0]] - pts[FACE_W[1]])
    if SKIN:
        face = soft(img.shape, [pts[OVAL]], [pts[i] for i in (EYE_L, EYE_R, BROW_L, BROW_R, LIPS_OUT)],
                    blur=int(fw * 0.06))
        ys, xs = np.nonzero(face > 0.01)
        if len(ys):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            roi = img[y0:y1, x0:x1]
            d = max(5, int(fw / 60))
            smooth = cv2.bilateralFilter(roi, d, 30, d)
            w = face[y0:y1, x0:x1, None] * SKIN
            img[y0:y1, x0:x1] = (roi * (1 - w) + smooth * w).astype(np.uint8)
    blush = np.zeros(img.shape[:2], np.float32)
    for c in CHEEKS:
        cv2.circle(blush, tuple(int(v) for v in pts[c]), int(fw * BLUSH_R), 1.0, -1)
    k = int(fw * BLUSH_R * 1.5) | 1
    tint(img, cv2.GaussianBlur(blush, (k, k), 0), BLUSH_RGB, BLUSH_ALPHA)
    tint(img, soft(img.shape, [pts[LIPS_OUT]], [pts[LIPS_IN]], blur=int(fw * 0.015)), LIP_RGB, LIP_ALPHA, LIP_DARKEN)


def frame_size(video):
    # размер уже повёрнутого кадра (телефонные видео хранят поворот в метаданных)
    png = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", "1", "-f", "image2pipe",
                          "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    h, w = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    return w, h


def apply(video):
    video = Path(video)
    out = CACHE / f"{key(video)}.makeup.mp4"
    part = out.with_suffix(".part.mp4")  # an interrupted run must not leave a half file that looks cached
    if out.exists():
        return out
    _, _, fps_str, fps, dur = probe(video)
    w, h = frame_size(video)
    mp, lm = landmarker()
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps_str}",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    # видео без потерь заметных глазу (CQ 16), звук копией — монтаж дальше режет этот файл как исходник
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", fps_str, "-i", "-", "-i", str(video), "-map", "0:v", "-map", "1:a?",
                            *venc(16), "-c:a", "copy", str(part)],
                           stdin=subprocess.PIPE)
    small = (DETECT_W, round(h * DETECT_W / w))
    prev, n, found = None, 0, 0
    while raw := dec.stdout.read(w * h * 3):
        img = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=np.ascontiguousarray(cv2.resize(img, small))), round(n * 1000 / fps))
        if res.face_landmarks:
            pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], np.float32)
            prev = pts if prev is None else prev + (pts - prev) * SMOOTH
            paint(img, prev)
            found += 1
        else:
            prev = None  # лицо пропало — не тянуть старые точки
        enc.stdin.write(img.tobytes())
        n += 1
        if n % 100 == 0:
            print(f"\r  makeup {n / fps:.0f}/{dur:.0f}s", end="", flush=True)
    enc.stdin.close()
    if dec.wait() or enc.wait():
        part.unlink(missing_ok=True)
        sys.exit("makeup: ffmpeg failed")
    part.replace(out)
    print(f"\r  makeup: лицо найдено в {found}/{n} кадрах -> {out}")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_file():
        sys.exit(__doc__)
    apply(sys.argv[1])
