# Reels / Shorts editing — talking head, 9:16

**Before every new video, re-read this whole file and apply "My editing rules".**
Write every correction the user makes as a separate rule under "My editing rules" (date, what, where it lives — constant/code in `montage.py`) and put it into the code right away. Write rules in the language the user writes in.
Talk to the user in their language. Speech in the videos is Russian by default; any other language goes through `--lang=xx`.

## Pipeline — `montage.py`

```
.venv\Scripts\python montage.py <video> [--music] [--makeup] [--denoise] [--dry-run] [--lang=en]
```

1. **Sound → speech islands.** `silencedetect`, pause > `PAUSE` = 0.4 s. The silence threshold is relative: `SILENCE_BELOW` = 25 dB under this recording's loud level (90th percentile) — a quiet stream mix (−46 dB) sits entirely under any fixed threshold. With `--music`, Demucs (htdemucs) first splits voice from music and everything below works on the clean voice.
2. **Transcription.** faster-whisper `large-v3`, word timestamps, only over the speech islands (otherwise Whisper stretches words over pauses). GPU float16 when there is an NVIDIA card (`has_gpu()`), CPU int8 otherwise. Typical Whisper hallucinations («Субтитры сделал…», «Продолжение следует…») are dropped. Language check first: if Whisper is ≥80% sure the speech is not `LANG`, the script stops (forced language makes it translate silently) — then run with `--lang=xx`. `FILLER_PROMPT` makes it write «ээ/мм» (Russian only).
3. **Retakes.** A phrase ≥60% of which (`TAKE_MATCH`, ≥2 words) reappears contiguously in one of the next 3 phrases is a failed take; the last attempt stays.
4. **Plan before rendering.** Printed: duration before/after, retakes with timecodes, fillers, how much went to pauses. `--dry-run` prints the plan and stops.
5. **Cuts.** Pauses > 0.4 s are cut, `PAD` = 0.15 s kept at speech edges; `XFADE` = 30 ms audio crossfade at each join. Sound without words (laugh, sigh) is kept. Fillers («ээ», «мм», «хм», `FILLER`) are cut inside phrases (Russian only). `--denoise` — voice noise reduction (`afftdn`). Cuts on the frame grid, audio length = video length; duration taken from the video stream (Telegram circles end before their audio).
6. **Picture.** 9:16 crop following the face frame by frame (MediaPipe Face Landmarker `TRACK_HZ` = 10 samples/s → `smooth_path`: gaps filled, median of 5, dead zone `CAM_DEAD` = 4% of the width so swaying doesn't move the camera, zero-phase gaussian `CAM_SIGMA` = 0.4 s; the crop position goes to ffmpeg per frame via `sendcmd`; it jumps only at cuts; a piece with no face keeps the previous camera) → 1080×1920; zoom centred on the face; 100% ↔ 108% zoom across cuts on a smootherstep curve (no jolt at start/end, `ZOOM_T` = 0.5 s); pieces ≥ `DRIFT_MIN` = 3 s creep in by `DRIFT` = 3% over their length and drop back at the cut (hidden by it); `OUTRO` = 1 s tail with fade to black.
7. **Captions.** Word-by-word `.ass`: Arial bold 110 px, centred, 2–3 words per line, current word yellow and popping to 112% in 80 ms (`POP`). Burned in.
8. **Output.** `out/<name>_edit.mp4` (H.264 — NVENC CQ 21, or libx264 CRF 21 without NVIDIA via `venc()`; AAC 192k, faststart), loudness −14 LUFS (gain + limiter at −1.5 dBFS, `normalize()`, gain capped at 40 dB `MAX_GAIN` so an empty take does not turn into hiss), plus `out/<name>.ass`, `.srt`, `.edl` (CMX 3600: import into DaVinci/Premiere for manual polishing, relink the clip by name). Cache: `out/cache/<name>.<size+mtime>.*` — the key includes size and date so two different `IMG_0001.mp4` never share a transcript (delete `*.words.json` after changing the model, `SILENCE_BELOW`, `PAUSE` or `FILLER_PROMPT`; stems `*.voice.wav`/`*.music.wav`).

Logic check without GPU or video: `.venv\Scripts\python test_montage.py`.

**Makeup** — `--makeup` (or standalone `makeup.py <video>`): MediaPipe Face Landmarker (model downloads to `out/cache/face_landmarker.task`) → lipstick, blush, skin smoothing (colour in LAB, a/b only) → `out/cache/<name>.<key>.makeup.mp4` with the original audio (written via `.part.mp4`, an interrupted run leaves no broken cache); the render cuts it instead of the source. Constants `LIP_*`, `BLUSH_*`, `SKIN`, `SMOOTH` in `makeup.py`; delete `*.makeup.mp4` after changing them.

**Gameplay Shorts** (not talking head) — `rl_short.py`: clip list with timecodes `CLIPS`, replays back to back, 16:9 full width over a blurred 9:16 fill; music is only the song's choruses (`CHORUSES`, bar-aligned, looped to the video length) with game sound under it (`GAME_VOL`, default 0.45, 0 = muted); goal cuts land on the song's beats (librosa; the clip start moves ≤ half a beat, the end — the goal — never moves), 60 fps, −14 LUFS → `out/rl_short.mp4`. The song is found by the `MUSIC_GLOB` mask (the path has «É» in another Unicode form and cannot be typed). Paths are the repository author's — change them for other clips.
**Feature demo videos** (16:9, slides/screens + voice-over) — `demo.py <script.json> [--lang ru|uk|en] [--voice DIR] [--design male] [--speed] [--tts omni|eleven]`: script JSON = segments of shots (`src`, `roi` to zoom into, `ss`/`crop_top` for clips) + narration per language (example `examples/robot_demo.json`); eased zoom into the roi, same-picture shots pan, others crossfade `XF` = 0.6 s, fade in/out; voice by OmniVoice (local, `omni_tts.py` worker in `.venv-tts` on Python 3.13 — it does not install on 3.14), each segment cloned from one fixed sample (`--voice` dir with ref.wav + ref.txt, default `out/voices/<design>-<lang>` designed once) so the voice never drifts; caption word times from Whisper on the generated audio mapped onto the script's spelling (`align`); ElevenLabs as the alternative (`secrets/elevenlabs.key`, script `"say"` for mispronounced words). Own voice: `make_ref.py <recording> <lang> <dir> [start]` cuts 7–14 s on quiet points + 0.15 s silence (a sample starting on a word tail makes every phrase start with it). Week material in `out/demo/<name>/`, voice samples in `secrets/voice/` (both out of git). Output `demo_<lang>.mp4` + `.srt` next to the script.
**YouTube upload** — `upload.py <video> --title ... [--tags a,b] [--privacy] [--publish-at ISO] [--category 20] [--dry-run]`: YouTube Data API v3, OAuth client `secrets/client_secret.json` (Desktop app), token `secrets/token.json`. Until Google audits the app, uploads are private.
**Stream highlights** — `fetch.py <url> [--top 10] [--pre 20] [--post 8] [--no-video]`: yt-dlp downloads only the audio → loudness spikes over the minute's background → downloads only those pieces (1080p60) to `out/streams/<id>/<start>.mp4` and prints `CLIPS` for `rl_short.py`.
**Video check** — `yt_check.py [id|url ...]` (no args — last 50 channel videos): rejections, processing errors, country blocks, 18+, "made for kids", Shorts length. Which track Content ID caught is only in Studio.
Environment: `.venv` (Python 3.14), dependencies in `requirements.txt`; ffmpeg from winget; for NVIDIA RTX 30/40/50 torch from the cu128 index (see README). Works without NVIDIA, slower.

## Known limits (check on a live recording)

- **Fillers:** cutting «ээ/мм» has not yet fired on real speech — `FILLER_PROMPT` may both miss and invent them. Read the `fillers:` line of the plan.
- **Retakes:** a natural repeat of a short phrase («то есть…», or a game character repeating the chosen dialogue line) within 3 phrases can be taken for a retake — check `retakes:` in `--dry-run`.
- **EDL:** import into DaVinci/Premiere not tried; timecodes are frame-exact (test).
- **Face framing:** follows one face (the first MediaPipe finds); with two people the crop may hop between them — active-speaker choice by mouth movement is not done yet. Crop moves in whole source pixels (`exact=1`): a very slow pan steps by ~2 output px. The zoom centre is one value per piece, so while the camera eases after a walking person the face sits up to ~10% off centre (robot test). Face times come from OpenCV's frame timestamps — a VFR source with a non-zero start time could shift the camera path against the render (not seen yet). A game cutscene switching cameras inside one piece is smoothed across the switch (no scene-cut detection yet). Tested on the drawn robot (`samples/robot.mp4`), not yet on a live recording.
- **Streams:** a small webcam in the corner is not found by the face model — the crop centres on the game UI. A "webcam on top, game below" layout would fit stream clips better.
- **VFR sources** (Telegram circles) render at their average fps (e.g. 29.01) — platforms accept it.

## Installed skills (`.claude/skills`, project level)

- **Remotion** (`remotion-*`, 12, remotion-dev/skills) — Remotion React compositions: create, captions, render, studio, maps. Instructions only; network — search in remotion.dev docs.
- **iart-ai/tiktok-video-skills** — `short-form-video` (hook → retention → loop, 9:16 safe areas), `caption-animation`, `lower-thirds`, `countdown-video`.
- **video-editing-skill** (6missedcalls) — ffmpeg bash scripts: trim, jumpcut, caption, overlay, speed. Transcription switched to faster-whisper (`scripts/transcribe_fw.py`, large-v3, ru, word timestamps, GPU or CPU) from the project `.venv`. `onboard.sh` removed — it installed the skill into `~/.claude`, outside the project.

Remotion and iart files live in `.agents/skills`, `.claude/skills` links to them (`npx skills add`).

## My editing rules

1. **2026-10-14 — no abrupt transitions between phrases.** 0.15 s of silence kept at cut edges (`PAD`).
2. **2026-10-14 — background music must transition smoothly.** Videos with music run with `--music` (rule 5), so the music is not cut at all. ~~0.3 s crossfade on every cut~~ — replaced by 30 ms per the project spec of 2026-10-14 (`XFADE`).
3. **2026-10-14 — smooth zoom on cuts.** Alternating 100%/108%, 0.5 s transition centred on the cut (`ZOOM`, `ZOOM_T`).
4. **2026-10-14 — smooth ending.** 1 s tail after the last phrase, fade to black and silence (`OUTRO`).
5. **2026-10-14 — if the music is one track, use it whole, don't cut it.** Run with `--music`: Demucs splits the voice off, only the voice is cut, the original track plays under it in one piece.
6. **2026-10-14 — gameplay Shorts: only the song's chorus; mute the game sound only on request.** By default game sound stays under the music (`GAME_VOL` 0.45); `GAME_VOL = 0` only when the user asks. Find choruses by lyrics (Whisper on the Demucs vocal stem) + by sound (chroma), cut on downbeats, chain choruses to the video length (`rl_short.py`, `CHORUSES`).
7. **2026-10-22 — демо фич из приложения: снимать кадр только после полной загрузки.** Ждать, пока подгрузятся имена, аватары и суммы; кадры со скелетонами, «Unknown fan» или $0 вместо данных переснимать. Процесс, не код: проверять скриншот перед сохранением кадра/записью клипа.
8. **2026-10-22 — тестовая «говорящая голова» — робот, не живое лицо.** Для проверок кадрирования/слежения использовать `samples/robot.mp4`: нарисованный робот (MediaPipe его видит) ходит по кадру 1920×1080, рот от громкости голоса. Генератор `make_robot.py <voice.wav> <out.mp4>`.
