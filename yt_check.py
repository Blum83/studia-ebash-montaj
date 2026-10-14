"""Что YouTube думает о видео: отказ/ошибка обработки, блокировки по странам, 18+, «для детей», попадёт ли в Shorts.

.venv\\Scripts\\python yt_check.py              # последние 50 видео канала (эфиры тоже)
.venv\\Scripts\\python yt_check.py <id|url> ...

Какой именно трек/правообладатель поймал Content ID — API не отдаёт, это только в Studio → «Ограничения».
"""
import re
import sys

from upload import youtube

REJECT = {
    "claim": "заявка правообладателя", "copyright": "нарушение авторских прав", "duplicate": "дубликат",
    "inappropriate": "неприемлемый контент", "legal": "юридическая причина", "length": "слишком длинное",
    "termsOfUse": "нарушение правил", "trademark": "товарный знак",
    "uploaderAccountClosed": "аккаунт закрыт", "uploaderAccountSuspended": "аккаунт заблокирован",
}
SHORTS_MAX = 180  # с


def vid(s):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([\w-]{11})", s)
    return m[1] if m else s


def seconds(iso):
    h, m, s = (int(x or 0) for x in re.fullmatch(r"P(?:\d+D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso).groups())
    return h * 3600 + m * 60 + s


def problems(v):
    st, cd = v["status"], v["contentDetails"]
    pd, fd = v.get("processingDetails", {}), v.get("fileDetails", {})
    out = []
    if st["uploadStatus"] in ("rejected", "failed", "deleted"):
        why = REJECT.get(st.get("rejectionReason"), st.get("rejectionReason") or st.get("failureReason", ""))
        out.append(f"❌ {st['uploadStatus']}: {why}")
    if pd.get("processingStatus") in ("failed", "terminated"):
        out.append(f"❌ обработка: {pd.get('processingFailureReason', pd['processingStatus'])}")
    elif pd.get("processingStatus") == "processing":
        out.append("⏳ ещё обрабатывается")
    rr = cd.get("regionRestriction", {})
    if rr.get("blocked"):
        out.append(f"⚠ заблокировано в {len(rr['blocked'])} странах: {', '.join(sorted(rr['blocked'])[:15])}"
                   f"{' …' if len(rr['blocked']) > 15 else ''} — обычно Content ID (музыка), детали в Studio")
    if "allowed" in rr:
        out.append(f"⚠ доступно только в: {', '.join(sorted(rr['allowed'])) or 'нигде'}")
    if cd.get("contentRating", {}).get("ytRating") == "ytAgeRestricted":
        out.append("⚠ 18+ (age restricted)")
    if st.get("madeForKids"):
        out.append("⚠ помечено «для детей»: нет комментариев и персональной рекламы")
    if st["privacyStatus"] != "public":
        when = f", публикация {st['publishAt']}" if st.get("publishAt") else ""
        out.append(f"· {st['privacyStatus']}{when}")
    vs = (fd.get("videoStreams") or [{}])[0]
    if vs.get("widthPixels") and vs["widthPixels"] < vs["heightPixels"] and seconds(cd["duration"]) > SHORTS_MAX:
        out.append(f"· вертикальное, но длиннее {SHORTS_MAX // 60} мин — в Shorts не попадёт")
    return out


def main(args):
    yt = youtube()
    if args:
        ids = [vid(a) for a in args]
    else:
        ch = yt.channels().list(part="contentDetails", mine=True).execute()["items"][0]
        uploads = ch["contentDetails"]["relatedPlaylists"]["uploads"]
        items = yt.playlistItems().list(part="contentDetails", playlistId=uploads, maxResults=50).execute()["items"]
        ids = [i["contentDetails"]["videoId"] for i in items]
    for i in range(0, len(ids), 50):
        res = yt.videos().list(part="snippet,status,contentDetails,processingDetails,fileDetails",
                               id=",".join(ids[i:i + 50])).execute()["items"]
        for v in res:
            p = problems(v)
            print(f"{' ' if any(x[0] in '❌⚠⏳' for x in p) else '✅'} {v['snippet']['title']}  https://youtu.be/{v['id']}")
            for line in p:
                print(f"     {line}")
        missing = set(ids[i:i + 50]) - {v["id"] for v in res}
        for m in missing:
            print(f"❓ {m}: не найдено (удалено или не твоё)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
