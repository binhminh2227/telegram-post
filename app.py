# app.py — Cloud Run “Worker style” (FastAPI)
# Endpoints: /setup, /status, /delete, /post-new
# - Lưu config & log bằng SQLite (file: /tmp/db.sqlite)
# - Đăng Telegram (album 1 bài duy nhất), caption = Title → Text → Fixed
# - Bearer cho /post-new (callback_bearer)
# - VPS sync (optional): VPS_BASE/VPS_BEARER
# - (Optional) Gemini: ENABLE_LLM=true + GEMINI_API_KEY

import os, json, asyncio, sqlite3, aiohttp, traceback, re
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.background import BackgroundTasks

# ---------- ENV ----------
PORT                 = int(os.environ.get("PORT", "8080"))
VPS_BASE             = os.environ.get("VPS_BASE", "").rstrip("/")
VPS_BEARER           = os.environ.get("VPS_BEARER", "")
CALLBACK_BEARER_DEF  = os.environ.get("CALLBACK_BEARER", "")

ENABLE_LLM           = os.environ.get("ENABLE_LLM", "false").lower() == "true"
GEMINI_API_KEY       = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL         = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE          = os.environ.get("GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta")

DB_PATH              = os.environ.get("DB_PATH", "/tmp/db.sqlite")
ALBUM_CAPTION_LIMIT  = 1024
TG_MAX_ALBUM         = 10

app = FastAPI(title="tg-worker-cloudrun")

# ---------- DB (SQLite: /tmp) ----------
def db_init():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS configs(
          channel TEXT PRIMARY KEY,
          payload TEXT NOT NULL
        );
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS logs(
          channel TEXT PRIMARY KEY,
          payload TEXT NOT NULL
        );
        """)
        con.commit()

def db_get_config(channel:str)->Optional[Dict[str,Any]]:
    with sqlite3.connect(DB_PATH) as con:
        cur=con.execute("SELECT payload FROM configs WHERE channel=?",(channel,))
        row=cur.fetchone()
        if not row: return None
        try: return json.loads(row[0])
        except: return None

def db_set_config(channel:str, payload:Dict[str,Any]):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT OR REPLACE INTO configs(channel,payload) VALUES(?,?)", (channel, json.dumps(payload, ensure_ascii=False)))
        con.commit()

def db_del_config(channel:str):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM configs WHERE channel=?", (channel,))
        con.execute("DELETE FROM logs WHERE channel=?", (channel,))
        con.commit()

def db_list_channels()->List[str]:
    with sqlite3.connect(DB_PATH) as con:
        cur=con.execute("SELECT channel FROM configs ORDER BY channel")
        return [r[0] for r in cur.fetchall()]

def db_set_log(channel:str, payload:Dict[str,Any]):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT OR REPLACE INTO logs(channel,payload) VALUES(?,?)", (channel, json.dumps(payload, ensure_ascii=False)))
        con.commit()

def db_get_log(channel:str)->Optional[Dict[str,Any]]:
    with sqlite3.connect(DB_PATH) as con:
        cur=con.execute("SELECT payload FROM logs WHERE channel=?",(channel,))
        row=cur.fetchone()
        if not row: return None
        try: return json.loads(row[0])
        except: return None

# ---------- Helpers ----------
def norm_channel(s:str)->str:
    if not s: return ""
    s = s.strip()
    m = re.match(r"^https?://t\.me/(?:s/)?([^/]+)", s, flags=re.I)
    if m: s = m.group(1)
    if s.startswith("@"): s = s[1:]
    return s

def json_resp(obj,status=200):
    return JSONResponse(obj,status_code=status)

def safe_str(v): return "" if v is None else str(v)

def split_caption(s:str, limit:int=4096):
    s = s or ""
    if len(s) <= limit: return {"head": s, "tails": []}
    head = s[:limit]; rest = s[limit:]; tails=[]
    while rest:
        tails.append(rest[:limit]); rest = rest[limit:]
    return {"head": head, "tails": tails}

def trim_album_caption(s:str):
    return s if len(s)<=ALBUM_CAPTION_LIMIT else (s[:ALBUM_CAPTION_LIMIT-1]+"…")

def first_non_empty_line(s:str)->str:
    for ln in (s or "").replace("\r\n","\n").split("\n"):
        t=ln.strip()
        if t: return t
    return ""

def strip_markdown_images(s:str)->str:
    s = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", s or "")
    s = re.sub(r"\[(?:img|image|photo)\]", "", s, flags=re.I)
    s = re.sub(r"\((?:img|image|photo)\)", "", s, flags=re.I)
    return s

def strip_links_preserve_lines(s:str)->str:
    s = strip_markdown_images(s or "")
    s = (s or "").replace("\r\n","\n")
    s = re.sub(r"https?://[^\s)>\]]+","",s, flags=re.I)
    s = "\n".join([ln.rstrip(" \t") for ln in s.split("\n")])
    s = re.sub(r"\n{3,}","\n\n",s)
    return s.strip()

def remove_boilerplate_lines(s:str)->str:
    bad = [
        r"^tl;?dr\b", r"^summary\b", r"^recap\b", r"^highlights?\b",
        r"^mmo101\b",
        r"^(source|nguồn|cre|credit|credits|follow|subscribe|join|contact|liên hệ|kết nối)\b",
        r"^(image|photo|ảnh|hình ảnh)\b",
        r"quick\s*recap"
    ]
    out=[]
    for ln in (s or "").replace("\r\n","\n").split("\n"):
        t=ln.strip()
        if not t: out.append(ln); continue
        if any(re.search(p,t,flags=re.I) for p in bad): 
            continue
        out.append(ln)
    res="\n".join(out)
    res=re.sub(r"\n{3,}","\n\n",res)
    return res.strip()

def dedupe_title_in_text(title:str, text:str)->str:
    if not title or not text: return text or ""
    lines=(text or "").replace("\r\n","\n").split("\n")
    idx=next((i for i,l in enumerate(lines) if l.strip()), -1)
    if idx>=0:
        a=re.sub(r"\s+"," ",title.strip().lower())
        b=re.sub(r"\s+"," ",lines[idx].strip().lower())
        if a and a==b: lines.pop(idx)
    res="\n".join(lines)
    res=re.sub(r"\n{3,}","\n\n",res)
    return res.strip()

def neutralize_slash_cmd(s:str)->str:
    t=s.lstrip()
    if t.startswith("/"): 
        # chèn zero width char để tránh /start
        return "\u200B"+s
    return s

def build_caption(title:str, text:str, fixed:str)->str:
    parts=[]
    if title: parts.append(title.strip())
    if text: parts.append(text.strip())
    if fixed: parts.append(fixed.strip())
    out="\n\n".join([p for p in parts if p]).strip()
    return neutralize_slash_cmd(out)

async def resolve_chat_id(token:str, username:str)->str:
    try:
        u = username if username.startswith("@") else "@"+username
        async with aiohttp.ClientSession() as s:
            async with s.get(f"https://api.telegram.org/bot{token}/getChat",
                             params={"chat_id": u}, timeout=15) as r:
                d=await r.json()
                _id=d.get("result",{}).get("id")
                return str(_id) if _id is not None else ""
    except: return ""

async def tg_send_text(token, chat_id, text):
    async with aiohttp.ClientSession() as s:
        async with s.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id":chat_id,"text":text}, timeout=30) as r:
            return await r.json()

async def tg_upload_photo(token, chat_id, bytes_data, filename, caption):
    form=aiohttp.FormData()
    form.add_field("chat_id", chat_id)
    form.add_field("photo", bytes_data, filename=filename, content_type="image/jpeg")
    if caption: form.add_field("caption", caption)
    async with aiohttp.ClientSession() as s:
        async with s.post(f"https://api.telegram.org/bot{token}/sendPhoto", data=form, timeout=120) as r:
            return await r.json()

async def tg_upload_video(token, chat_id, bytes_data, filename, caption):
    form=aiohttp.FormData()
    form.add_field("chat_id", chat_id)
    form.add_field("video", bytes_data, filename=filename, content_type="video/mp4")
    if caption: form.add_field("caption", caption)
    async with aiohttp.ClientSession() as s:
        async with s.post(f"https://api.telegram.org/bot{token}/sendVideo", data=form, timeout=300) as r:
            return await r.json()

async def fetch_to_blob(url:str)->Tuple[bytes,str,str]:
    async with aiohttp.ClientSession() as s:
        async with s.get(url, timeout=300) as r:
            if r.status!=200: raise RuntimeError(f"media GET {r.status}")
            ct=r.headers.get("content-type","application/octet-stream")
            ext=(ct.split("/")[1] if "/" in ct else "bin").split("+")[0]
            data=await r.read()
            return data, f"file.{ext}", ct

async def send_media_set(token:str, chat_id:str, cap:Dict[str,Any], filesOrdered, urlsOrdered):
    # Prefetch URLs
    url_blobs=[]
    for u in urlsOrdered or []:
        try:
            data, fname, ct = await fetch_to_blob(u["url"])
            kind = "video" if str(u.get("mime","")).lower().startswith("video/") else "photo"
            url_blobs.append({"kind":kind,"bytes":data,"name":fname})
        except Exception as e:
            print("prefetch error:", u.get("url"), e)

    # Merge all
    all_items = []
    for f in filesOrdered or []:
        k = "video" if str(f.get("type","")).startswith("video/") else ("photo" if str(f.get("type","")).startswith("image/") else "other")
        if k in ("photo","video"):
            all_items.append({"kind":k,"bytes":f["bytes"],"name":f.get("name") or f"{k}.bin"})
    for b in url_blobs:
        all_items.append(b)

    if not all_items:
        head = cap["head"]
        resp = await tg_send_text(token, chat_id, head)
        # tails với text-only
        for t in cap.get("tails",[]):
            if t.strip(): await tg_send_text(token, chat_id, t)
        return {"ok": bool(resp.get("ok")), "method":"text-only"}

    if len(all_items)>=2:
        used = all_items[:TG_MAX_ALBUM]
        form=aiohttp.FormData()
        media=[]
        for i,it in enumerate(used):
            field=f"attach{i}"
            form.add_field(field, it["bytes"], filename=it["name"])
            media.append({
                "type":"video" if it["kind"]=="video" else "photo",
                "media":f"attach://{field}",
                "caption": trim_album_caption(cap["head"]) if i==0 else None
            })
        form.add_field("chat_id",chat_id)
        form.add_field("media",json.dumps(media, ensure_ascii=False))
        async with aiohttp.ClientSession() as s:
            async with s.post(f"https://api.telegram.org/bot{token}/sendMediaGroup", data=form, timeout=300) as r:
                d=await r.json()
        return {"ok": bool(d.get("ok")), "method":"media-group","sent_count":len(used)}

    it = all_items[0]
    if it["kind"]=="video":
        d = await tg_upload_video(token, chat_id, it["bytes"], it["name"] or "video.mp4", cap["head"])
        ok = bool(d.get("ok"))
        # tails allowed
        for t in cap.get("tails",[]):
            if t.strip(): await tg_send_text(token, chat_id, t)
        return {"ok": ok, "method":"video-binary"}
    else:
        d = await tg_upload_photo(token, chat_id, it["bytes"], it["name"] or "photo.jpg", cap["head"])
        ok = bool(d.get("ok"))
        for t in cap.get("tails",[]):
            if t.strip(): await tg_send_text(token, chat_id, t)
        return {"ok": ok, "method":"photo-binary"}

# ---------- LLM (Gemini optional) ----------
async def call_gemini_clean(sys_prompt:str, text:str)->Tuple[str,Dict[str,Any]]:
    if not ENABLE_LLM or not GEMINI_API_KEY:
        return text, {"ok": False, "status": 0, "error": "LLM disabled or no key"}
    url = f"{GEMINI_BASE}/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents":[
            {"role":"user","parts":[{"text": f"{sys_prompt}\n\n---\n\n{text or '(empty)'}"}]}
        ],
        "generationConfig":{"temperature":0.3}
    }
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=body, timeout=60) as r:
                data = await r.json()
                if r.status!=200:
                    return text, {"ok": False, "status": r.status, "error": data}
                out = (data.get("candidates",[{}])[0]
                           .get("content",{})
                           .get("parts",[{"text":""}])[0]
                           .get("text","")).strip()
                return (out or text), {"ok": True, "status": 200}
    except Exception as e:
        return text, {"ok": False, "status": 0, "error": str(e)}

SYS_PROMPT = (
    "Rewrite the input post to be clearer and more concise, but KEEP the original language.\n"
    "Preserve layout and line breaks. Remove ALL URLs.\n"
    "Do NOT include source captions/credits/call-to-action/promotional tags.\n"
    "Output ONLY the cleaned content. No emojis, no extra commentary."
)

# ---------- Schemas ----------
class Target(BaseModel):
    tg_bot_token: str
    target_chat_id: Optional[str] = ""
    target_channel: Optional[str] = ""

class SetupBody(BaseModel):
    channel: Optional[str] = ""
    source: Optional[str] = ""
    targets: List[Target]
    fixed_caption: Optional[str] = ""
    callback_bearer: Optional[str] = ""
    gpt_prompt: Optional[str] = ""  # forwarded to VPS (optional)

# ---------- Routes ----------
@app.post("/setup")
async def setup(body: SetupBody):
    ch = norm_channel(body.source or body.channel or "")
    if not ch: raise HTTPException(400, "Need 'channel' or 'source'")
    # resolve chat ids if needed
    resolved=[]
    for t in body.targets:
        tok=t.tg_bot_token.strip()
        chat_id=(t.target_chat_id or "").strip()
        uname=(t.target_channel or "").strip()
        if not tok or (not chat_id and not uname): 
            continue
        if not chat_id and uname:
            cid = await resolve_chat_id(tok, uname)
            if not cid: continue
            chat_id = cid
        resolved.append({"tg_bot_token":tok, "target_chat_id":chat_id, "target_channel":uname})
    if not resolved: raise HTTPException(400, "No valid targets")

    cfg = db_get_config(ch) or {}
    cfg.update({
        "channel": ch,
        "targets": resolved,
        "fixed_caption": body.fixed_caption or "",
        "callback_bearer": (body.callback_bearer or CALLBACK_BEARER_DEF or "")
    })
    db_set_config(ch, cfg)

    # VPS sync (optional)
    vps_added=False
    if VPS_BASE:
        try:
            src=f"https://t.me/s/{ch}"
            headers={"Authorization":f"Bearer {VPS_BEARER}"} if VPS_BEARER else {}
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{VPS_BASE}/channel", params={
                    "chanel": ch,
                    "source": src,
                    "prompt": (body.gpt_prompt or "")
                }, headers=headers, timeout=15) as r:
                    vps_added = r.status==200
        except: vps_added=False

    return json_resp({"ok": True, "saved": cfg, "vps_added": vps_added})

@app.get("/delete")
async def delete(channel: Optional[str]=None, source: Optional[str]=None):
    ch = norm_channel(source or channel or "")
    if not ch: raise HTTPException(400, "Missing channel/source")
    cfg = db_get_config(ch)
    if cfg and VPS_BASE:
        try:
            headers={"Authorization":f"Bearer {VPS_BEARER}"} if VPS_BEARER else {}
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{VPS_BASE}/delete", params={"chanel": ch}, headers=headers, timeout=15) as r:
                    _ = r.status
        except: pass
    db_del_config(ch)
    return json_resp({"ok": True, "deleted": ch})

@app.get("/status")
async def status(channel: Optional[str]=None, source: Optional[str]=None):
    ch = norm_channel(source or channel or "")
    if not ch:
        chans = db_list_channels()
        return json_resp({"ok": True, "total_channels": len(chans), "sample": chans[:20]})
    cfg = db_get_config(ch)
    log = db_get_log(ch)
    return json_resp({"ok": True, "channel": ch, "config": cfg, "last_post": log})

@app.post("/post-new")
async def post_new(req: Request):
    # Bearer check after we know channel (from payload)
    binFiles=[]  # [{'name','type','bytes'}]
    meta=None

    ctype = req.headers.get("content-type","").lower()
    if "multipart/form-data" in ctype:
        form = await req.form()
        jsonPart = form.get("json")
        if not jsonPart: raise HTTPException(400, "multipart needs 'json'")
        try:
            meta = json.loads(jsonPart if isinstance(jsonPart,str) else await jsonPart.read())
        except:
            raise HTTPException(400, "invalid json in 'json' field")
        for k,v in form.items():
            if not str(k).startswith("media"): continue
            if isinstance(v, UploadFile):
                data = await v.read()
                binFiles.append({"name": v.filename or k, "type": v.content_type or "application/octet-stream", "bytes": data})
    else:
        try:
            meta = await req.json()
        except:
            raise HTTPException(400, "invalid json body")

    ch = norm_channel(meta.get("channel") or meta.get("source") or meta.get("ch") or "")
    if not ch: raise HTTPException(400, "payload missing channel")
    cfg = db_get_config(ch)
    if not cfg: raise HTTPException(400, "channel not set up")

    # Bearer
    cb_secret = cfg.get("callback_bearer") or CALLBACK_BEARER_DEF or ""
    if cb_secret:
        auth = req.headers.get("authorization","")
        if not (auth.startswith("Bearer ") and auth[7:]==cb_secret):
            raise HTTPException(401, "bad bearer")

    # Extract message + media
    title = safe_str(meta.get("title")) or safe_str(meta.get("message",{}).get("title")) \
        or first_non_empty_line(
            safe_str(meta.get("message",{}).get("text")) or safe_str(meta.get("text")) \
            or safe_str(meta.get("caption")) or safe_str(meta.get("content")) or ""
        )
    text  = safe_str(meta.get("message",{}).get("text")) or safe_str(meta.get("text")) \
        or safe_str(meta.get("caption")) or safe_str(meta.get("content")) or ""

    raw_media = []
    raw_media += meta.get("media",[]) if isinstance(meta.get("media"), list) else []
    raw_media += meta.get("message",{}).get("media",[]) if isinstance(meta.get("message",{}).get("media",[]), list) else []

    url_media = []
    for m in raw_media:
        url = m.get("url") or m.get("src") or m.get("file_url") or m.get("link") or m.get("file")
        mime = m.get("mime") or m.get("type") or m.get("content_type") or ""
        if url:
            url_media.append({"url": url, "mime": mime})

    # Clean
    title = strip_links_preserve_lines(title).split("\n")[0].strip()
    cleaned = remove_boilerplate_lines(strip_links_preserve_lines(text))
    # LLM (optional)
    llm_info=None
    if ENABLE_LLM and GEMINI_API_KEY and cleaned.strip():
        cleaned, llm_info = await call_gemini_clean(SYS_PROMPT, cleaned)

    cleaned = strip_links_preserve_lines(cleaned)
    cleaned = remove_boilerplate_lines(cleaned)
    cleaned = dedupe_title_in_text(title, cleaned)
    fixed = strip_links_preserve_lines(cfg.get("fixed_caption",""))
    caption_full = build_caption(title, cleaned, fixed)
    cap = split_caption(caption_full, 4096)

    # Prepare "ordered" arrays
    filesOrdered = []
    for f in binFiles:
        kind = "video" if (f["type"] or "").startswith("video/") else ("photo" if (f["type"] or "").startswith("image/") else "other")
        if kind in ("photo","video"): filesOrdered.append({"kind":kind, **f})
    urlsOrdered = []
    for u in url_media:
        kind = "video" if (u.get("mime","").lower().startswith("video/")) else ("photo" if u.get("mime","").lower().startswith("image/") else "other")
        if kind in ("photo","video"): urlsOrdered.append({"kind":kind, "url":u["url"], "mime":u.get("mime","application/octet-stream")})

    # Send to all targets
    results=[]
    for t in cfg.get("targets",[]):
        tok = t.get("tg_bot_token")
        chat_id = t.get("target_chat_id")
        if not tok or not chat_id:
            results.append({"ok":False,"error":"invalid_target","target":t}); continue
        r = await send_media_set(tok, chat_id, cap, filesOrdered, urlsOrdered)
        results.append({"target_chat_id":chat_id, "method": r.get("method"), "ok": r.get("ok", False)})

    # Log
    log = {
        "at": __import__("datetime").datetime.utcnow().isoformat()+"Z",
        "channel": ch,
        "caption_head_len": len(cap["head"]),
        "tails_count": len(cap["tails"]),
        "files_count": len(filesOrdered),
        "urls_count": len(urlsOrdered),
        "results": results
    }
    if llm_info: log["gemini"]=llm_info
    db_set_log(ch, log)

    return json_resp({"ok": all(x.get("ok") for x in results), "results": results, "gemini": llm_info})

@app.get("/")
async def root():
    return json_resp({"ok": True, "hint": "Use /setup, /status, /delete, /post-new"})

# ---------- Startup ----------
@app.on_event("startup")
async def _startup():
    db_init()
