"""Загрузка ролика на YouTube (Shorts — это обычное видео: вертикальное и ≤ 3 мин).

.venv\\Scripts\\python upload.py out/x_edit.mp4 --title "..." [--desc ...] [--tags a,b]
    [--privacy private|unlisted|public] [--publish-at 2026-10-15T18:00:00+03:00]
    [--category 20] [--dry-run]

Один раз: secrets/client_secret.json (OAuth client «Desktop app» из Google Cloud).
При первом запуске откроется браузер, токен сохранится в secrets/token.json.
Пока приложение не прошло аудит Google, API принудительно ставит видео в private.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

SECRETS = Path(__file__).parent / "secrets"
CLIENT = SECRETS / "client_secret.json"
TOKEN = SECRETS / "token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly"]


def body(title, desc, tags, privacy, publish_at, category):
    status = {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}
    if publish_at:
        datetime.fromisoformat(publish_at)  # падает на кривой дате до загрузки
        status |= {"privacyStatus": "private", "publishAt": publish_at}  # API требует private
    return {
        "snippet": {"title": title[:100], "description": desc, "tags": tags, "categoryId": str(category)},
        "status": status,
    }


def youtube():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(TOKEN) if TOKEN.exists() else None
    if creds and not creds.has_scopes(SCOPES):
        creds = None  # токен со старым набором прав — заново вход в браузере
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    elif not creds or not creds.valid:
        if not CLIENT.exists():
            raise SystemExit(f"Нет {CLIENT} — скачай OAuth client (Desktop app) из Google Cloud.")
        creds = InstalledAppFlow.from_client_secrets_file(CLIENT, SCOPES).run_local_server(port=0)
    TOKEN.write_text(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def upload(path, meta):
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=8 << 20, resumable=True)
    req = youtube().videos().insert(part="snippet,status", body=meta, media_body=media)
    resp = None
    while resp is None:
        st, resp = req.next_chunk()
        if st:
            print(f"  {st.progress():.0%}")
    return resp["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--title")
    ap.add_argument("--desc", default="")
    ap.add_argument("--tags", default="")
    ap.add_argument("--privacy", default="private", choices=["private", "unlisted", "public"])
    ap.add_argument("--publish-at", help="ISO-время с поясом; видео станет public в этот момент")
    ap.add_argument("--category", default=22, type=int, help="22 People & Blogs, 20 Gaming")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not a.video.exists():
        raise SystemExit(f"Нет файла {a.video}")
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    meta = body(a.title or a.video.stem, a.desc, tags, a.privacy, a.publish_at, a.category)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if a.dry_run:
        return
    vid = upload(a.video, meta)
    print(f"Готово: https://youtube.com/shorts/{vid}")


if __name__ == "__main__":
    main()
