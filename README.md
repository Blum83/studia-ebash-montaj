# studia-ebash-montaj

[Русский](README.ru.md) · **English**

Talking-head Reels / Shorts (9:16) edited in one command — or by plain words in Claude Code.

Drop in a raw video → get a finished 1080×1920 clip:

- **pauses** longer than 0.4 s and **failed takes** are cut (a phrase started several times keeps only the last attempt);
- **fillers** ("uh", "um" — Russian «ээ», «мм») are cut;
- **word-by-word captions** in the centre, 2–3 words per line, the current word highlighted with a small pop;
- **the 9:16 frame follows the face**, smooth zoom on cuts, fade to black at the end;
- **loudness −14 LUFS**, what Reels/Shorts/TikTok normalise to;
- an `.srt` and an `.edl` next to it — open the edit in DaVinci Resolve / Premiere and fine-tune by hand.

Before rendering it prints a plan: what will be cut and why. `--dry-run` shows only the plan.

## Languages

Speech is **Russian by default**. Any language Whisper knows works with `--lang=xx` (`en`, `uk`, `de`, …).
Without the flag, speech in another language stops the run — otherwise Whisper would silently translate it.
Filler cutting is Russian-only for now; in other languages fillers simply stay.
The project instructions for Claude ([`CLAUDE.md`](CLAUDE.md)) and the script messages are in Russian — Claude reads them in any language you write to it.

---

## Requirements

| | |
|---|---|
| Windows 10/11 | tested on Windows 11; Linux/macOS should work but are untested |
| NVIDIA GPU (RTX 30/40/50) | a couple of minutes of video takes ~1 min. **Works without one too**, several times slower (CPU) |
| ~10 GB of disk | models download on first run: Whisper large-v3 ~3 GB, PyTorch ~3 GB |
| [Claude Code](https://claude.com/claude-code) | optional — to edit by words. The scripts run without it |

## Install (once, ~15 minutes)

Open PowerShell and run in order.

**1. Tools** (skip what you already have):

```powershell
winget install Git.Git Python.Python.3.14 Gyan.FFmpeg
```

Close and reopen PowerShell, otherwise `ffmpeg` is not found. Check: `ffmpeg -version`.

**2. Project:**

```powershell
git clone https://github.com/Blum83/studia-ebash-montaj
cd studia-ebash-montaj
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
```

**3. PyTorch for the GPU** (used to separate voice from music):

```powershell
.venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

No NVIDIA card — the same without `--index-url ...` (installs the CPU build).

**4. Everything else:**

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

**5. Check** (no video, no GPU, a few seconds):

```powershell
.venv\Scripts\python test_montage.py
```

It should print `ok` and `ok makeup`.

## Usage

### By words in Claude Code (easier)

Open the project folder in Claude Code and write, for example:

> edit `C:\Users\me\Videos\raw.mp4`

Claude reads [`CLAUDE.md`](CLAUDE.md) (the whole pipeline and the style rules), shows the plan and renders. Then correct it in words: "bigger captions", "don't cut pauses that hard", "add music". **Every correction is written into `CLAUDE.md` as a rule** ("My editing rules") and applied to every next video — that is how the project learns your style.

> `CLAUDE.md` already contains the repository author's rules (smooth transitions, zoom on cuts, …). Keep them or delete them — it is a style, not a requirement.

### By command (no Claude)

```powershell
.venv\Scripts\python montage.py C:\path\raw.mp4 --dry-run   # plan only: what will be cut
.venv\Scripts\python montage.py C:\path\raw.mp4             # edit
```

Output goes to `out\`: `raw_edit.mp4`, `raw.ass` (captions), `raw.srt`, `raw.edl`.

| Flag | When |
|---|---|
| `--music` | background music in the video: a neural net splits the voice from it, only the voice is cut, the track plays on uncut |
| `--denoise` | noisy room, fan, street |
| `--makeup` | light makeup on the face (lipstick, blush, skin smoothing) |
| `--lang=en` | speech is not Russian (see [Languages](#languages)) |
| `--dry-run` | show the plan, don't render |

Settings (pauses, caption size and colour, zoom…) are constants at the top of [`montage.py`](montage.py), each with a comment.

## Other scripts

| File | What it does |
|---|---|
| [`rl_short.py`](rl_short.py) | a Short from game clips set to music: clips back to back, cuts on the song's beat, chorus only. **The author's example** — change the clip paths (`SRC`, `CLIPS`) and the song (`MUSIC_GLOB`) to yours |
| [`fetch.py`](fetch.py) | finds the loud moments (shouts, reactions) in a YouTube stream recording and downloads only those pieces |
| [`upload.py`](upload.py) | upload to YouTube. Needs your own Google OAuth client in `secrets\client_secret.json` (instructions at the top of the file) |
| [`yt_check.py`](yt_check.py) | what YouTube thinks of your videos: country blocks, "made for kids", will it get into Shorts |
| [`makeup.py`](makeup.py) | makeup on its own, without editing |

## Skills for Claude Code

`.claude/skills` holds ready skills Claude picks up by itself:

- **Remotion** — animated titles, transitions, graphics in code (when ffmpeg is not enough);
- **short-form-video, caption-animation, lower-thirds** — hooks, retention, 9:16 safe areas, name plates;
- **video-editing-skill** — simple ffmpeg bash tools (trim, speed up, text overlays).

## Troubleshooting

- **Cut too much / left pauses** — run `--dry-run` first and read the `retakes:` and `fillers:` lines. A natural repeat ("I mean… I mean") is sometimes taken for a failed take. The silence threshold is `SILENCE_BELOW` in `montage.py`.
- **Changed settings, same result** — the transcript is cached in `out\cache\*.words.json`, delete it.
- **`speech is 'en' ... not 'ru'`** — the speech is not Russian, add `--lang=en`.
- **`ffmpeg` not found** — restart PowerShell after installing.
- **Very slow** — no NVIDIA card, or PyTorch with CUDA did not install (step 3). Check: `.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"` → `True`.
