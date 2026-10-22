"""OmniVoice worker for demo.py — runs in .venv-tts (Python 3.13; OmniVoice does not install on 3.14).
  omni_tts.py <jobs.json> <lang> <design> <voice_dir>   jobs [{text, out, speed}] -> wav files in the voice_dir voice
  omni_tts.py - <lang> <design> <voice_dir> <text>      design a new voice from <design> ("male", "female, low pitch")
                                                        and keep it as voice_dir/ref.wav + ref.txt
Cloning from one fixed sample keeps the voice identical across segments and weeks."""
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from omnivoice import OmniVoice

jobs, lang, design, voice = sys.argv[1:5]
voice = Path(voice)
m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                              dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
if jobs == "-":
    text = sys.argv[5]
    sf.write(voice / "ref.wav", m.generate(text=text, language=lang, instruct=design)[0].squeeze(), m.sampling_rate)
    (voice / "ref.txt").write_text(text, encoding="utf-8")
    sys.exit()
prompt = m.create_voice_clone_prompt(ref_audio=str(voice / "ref.wav"),
                                     ref_text=(voice / "ref.txt").read_text(encoding="utf-8"))
for j in json.loads(Path(jobs).read_text(encoding="utf-8")):
    a = m.generate(text=j["text"], language=lang, voice_clone_prompt=prompt, speed=j.get("speed"))[0]
    sf.write(j["out"], a.squeeze(), m.sampling_rate)
    print("voiced", Path(j["out"]).name, flush=True)
