#!/usr/bin/env python3
"""FastAPI entry-point for the Research Prerequisite Agent."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .llm_logging import append_log, create_log_path
from .research_prereq import ResearchPrerequisiteAgent


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent / "reports.db"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# 模块级单例连接，避免每次请求重新建立连接的开销
_db_connection: sqlite3.Connection | None = None
_db_lock = threading.Lock()


def get_db() -> sqlite3.Connection:
    global _db_connection
    if _db_connection is None:
        with _db_lock:
            if _db_connection is None:
                conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")  # 允许并发读写
                conn.execute("PRAGMA synchronous=NORMAL")  # 减少 fsync 次数
                _db_connection = conn
    return _db_connection


@contextmanager
def db_conn():
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _migrate_db():
    """Add columns introduced after initial schema creation."""
    with db_conn() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_configs)")}
        if "report_language" not in cols:
            conn.execute(
                "ALTER TABLE llm_configs ADD COLUMN report_language TEXT NOT NULL DEFAULT 'zh-CN'"
            )


def init_db():
    with db_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS llm_configs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT    NOT NULL,
                provider        TEXT    NOT NULL DEFAULT 'OpenAI',
                api_key         TEXT    NOT NULL,
                base_url        TEXT    NOT NULL DEFAULT '',
                model_name      TEXT    NOT NULL,
                report_language TEXT    NOT NULL DEFAULT 'zh-CN',
                created_at      TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                topic       TEXT    NOT NULL,
                format_style TEXT   NOT NULL DEFAULT 'IEEE',
                version     TEXT    NOT NULL DEFAULT '1.0',
                file_path   TEXT    NOT NULL DEFAULT '',
                markdown    TEXT    NOT NULL DEFAULT '',
                round1      TEXT,
                round2      TEXT,
                round3      TEXT,
                status      TEXT    NOT NULL DEFAULT 'pending',
                error       TEXT,
                created_at  TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id   INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
           role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
         );
        CREATE TABLE IF NOT EXISTS background_jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id   INTEGER, -- 可空，任务完成后关联生成的报告
            topic       TEXT    NOT NULL,
            format_style TEXT   NOT NULL,
            status      TEXT    NOT NULL DEFAULT 'running', -- running, done, error
            error       TEXT,
            created_at  TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS job_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      INTEGER NOT NULL REFERENCES background_jobs(id) ON DELETE CASCADE,
            status      TEXT,
            message     TEXT,
            created_at  TEXT    NOT NULL
        );
        """)
        # 在线迁移：如果旧表没有 status/error 字段，添加它们
        cursor = conn.execute("PRAGMA table_info(reports)")
        columns = [row[1] for row in cursor.fetchall()]
        if "status" not in columns:
            conn.execute(
                "ALTER TABLE reports ADD COLUMN status TEXT NOT NULL DEFAULT 'done'"
            )
        if "error" not in columns:
            conn.execute("ALTER TABLE reports ADD COLUMN error TEXT")

        # 在线迁移：为 background_jobs 添加 LLM 配置字段
        cursor = conn.execute("PRAGMA table_info(background_jobs)")
        job_columns = [row[1] for row in cursor.fetchall()]
        if "llm_api_key" not in job_columns:
            conn.execute("ALTER TABLE background_jobs ADD COLUMN llm_api_key TEXT")
        if "llm_model" not in job_columns:
            conn.execute("ALTER TABLE background_jobs ADD COLUMN llm_model TEXT")
        if "llm_base_url" not in job_columns:
            conn.execute("ALTER TABLE background_jobs ADD COLUMN llm_base_url TEXT")


init_db()
_migrate_db()

# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def _next_version(topic: str) -> tuple[str, Path]:
    """Compute next semantic version for this topic and return (version_str, file_path)."""
    safe = re.sub(r"[^\w\u4e00-\u9fff]+", "_", topic)[:40]
    with db_conn() as conn:
        row = conn.execute(
            "SELECT version FROM reports WHERE topic=? ORDER BY id DESC LIMIT 1",
            (topic,),
        ).fetchone()
    if row is None:
        version = "1.0"
    else:
        major, minor = row["version"].split(".")
        version = f"{int(major)}.{int(minor) + 1}"
    filename = f"{version}_{safe}_report.md"
    return version, REPORTS_DIR / filename


def _save_report_file(path: Path, markdown: str):
    path.write_text(markdown, encoding="utf-8")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Research Prerequisite API",
    description="Generate research prerequisite reports using LLM + web search.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LLMConfigCreate(BaseModel):
    name: str
    provider: str = "OpenAI"
    api_key: str = ""
    base_url: str = ""
    model_name: str
    report_language: str = "zh-CN"


class LLMConfigUpdate(BaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    report_language: Optional[str] = None


class LLMConfigItem(BaseModel):
    id: int
    name: str
    provider: str
    api_key: str
    base_url: str
    model_name: str
    report_language: str
    created_at: str


class ReportRequest(BaseModel):
    topic: str
    format_style: Optional[str] = "IEEE"
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    report_language: Optional[str] = "zh-CN"


class ReportResponse(BaseModel):
    id: int
    topic: str
    version: str
    file_path: str
    markdown: str
    round1: dict
    round2: dict
    round3: dict
    status: str
    error: Optional[str] = None
    created_at: str


class HistoryItem(BaseModel):
    id: int
    is_job: bool = False
    job_id: Optional[int] = None
    report_id: Optional[int] = None
    topic: str
    version: str
    file_path: str
    format_style: str
    created_at: str
    status: str = "done"
    error: Optional[str] = None
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None


class MessageItem(BaseModel):
    id: int
    report_id: int
    role: str
    content: str
    created_at: str


class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None


class RegenerateRequest(BaseModel):
    instruction: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None


class RetryJobRequest(BaseModel):
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: build agent
# ---------------------------------------------------------------------------


def _make_agent(
    api_key: str,
    model: Optional[str],
    base_url: Optional[str],
    log_path: Optional[str] = None,
    progress_callback=None,
) -> ResearchPrerequisiteAgent:
    kwargs: dict = {"api_key": api_key, "timeout": 120}
    if model:
        kwargs["model"] = model
    if base_url:
        kwargs["base_url"] = base_url
    if log_path:
        kwargs["log_path"] = log_path
    if progress_callback:
        kwargs["progress_callback"] = progress_callback
    return ResearchPrerequisiteAgent(**kwargs)


def _require_api_key(api_key: Optional[str], base_url: Optional[str] = None) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    # Ollama / 本地 OpenAI-Compatible 服务不需要真实 key
    if not key:
        bu = (base_url or "").lower()
        if "localhost" in bu or "127.0.0.1" in bu or "::1" in bu:
            return "ollama"
        raise HTTPException(
            status_code=400, detail="api_key 必填（或设置 OPENAI_API_KEY 环境变量）"
        )
    return key


def _run_report_job(
    job_id: int,
    topic: str,
    format_style: str,
    api_key: str,
    model: Optional[str],
    base_url: Optional[str],
    report_topic: Optional[str] = None,
    post_save_message: Optional[str] = None,
    report_language: Optional[str] = "zh-CN",
):
    report_topic = report_topic or topic
    log_path = str(create_log_path(f"job_{job_id}_{report_topic}"))

    def worker():
        def progress_callback(event: dict):
            with db_conn() as conn:
                conn.execute(
                    "INSERT INTO job_events (job_id, status, message, created_at) VALUES (?, ?, ?, ?)",
                    (
                        job_id,
                        event.get("status"),
                        event.get("message", ""),
                        datetime.utcnow().isoformat(),
                    ),
                )
                conn.execute(
                    "UPDATE background_jobs SET updated_at=? WHERE id=?",
                    (datetime.utcnow().isoformat(), job_id),
                )

        agent = _make_agent(
            api_key,
            model,
            base_url,
            log_path=log_path,
            progress_callback=progress_callback,
        )

        try:
            for step in agent.generate_report(
                topic,
                format_style=format_style,
                report_language=report_language or "zh-CN",
            ):
                if step.get("status") == "done":
                    result = step["result"]
                    version, file_path = _next_version(report_topic)
                    _save_report_file(file_path, result["markdown"])

                    now_done = datetime.utcnow().isoformat()
                    with db_conn() as conn:
                        cur = conn.execute(
                            """INSERT INTO reports (topic, format_style, version, file_path, markdown, round1, round2, round3, created_at, status)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'done')""",
                            (
                                report_topic,
                                format_style,
                                version,
                                str(file_path),
                                result["markdown"],
                                json.dumps(
                                    result.get("planner", {}), ensure_ascii=False
                                ),
                                json.dumps(
                                    result.get("synthesizer", {}), ensure_ascii=False
                                ),
                                json.dumps(
                                    result.get("critic", {}), ensure_ascii=False
                                ),
                                now_done,
                            ),
                        )
                        report_id = cur.lastrowid
                        if post_save_message:
                            conn.execute(
                                "INSERT INTO messages (report_id, role, content, created_at) VALUES (?,?,?,?)",
                                (report_id, "system", post_save_message, now_done),
                            )
                        conn.execute(
                            "UPDATE background_jobs SET status='done', report_id=?, error=NULL, updated_at=? WHERE id=?",
                            (report_id, now_done, job_id),
                        )
                        conn.execute(
                            "INSERT INTO job_events (job_id, status, message, created_at) VALUES (?, ?, ?, ?)",
                            (
                                job_id,
                                "done",
                                json.dumps({"report_id": report_id}),
                                now_done,
                            ),
                        )
        except Exception as exc:
            now_err = datetime.utcnow().isoformat()
            with db_conn() as conn:
                conn.execute(
                    "UPDATE background_jobs SET status='error', error=?, updated_at=? WHERE id=?",
                    (str(exc), now_err, job_id),
                )
                conn.execute(
                    "INSERT INTO job_events (job_id, status, message, created_at) VALUES (?, ?, ?, ?)",
                    (job_id, "error", str(exc), now_err),
                )

    threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# LLM Config CRUD
# ---------------------------------------------------------------------------


@app.get("/api/llm-configs", response_model=List[LLMConfigItem])
async def list_llm_configs():
    """列出所有已保存的 LLM 配置。"""
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, provider, api_key, base_url, model_name, report_language, created_at FROM llm_configs ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/llm-configs", response_model=LLMConfigItem)
async def create_llm_config(cfg: LLMConfigCreate):
    """保存一条新的 LLM 配置。"""
    now = datetime.utcnow().isoformat()
    with db_conn() as conn:
        cur = conn.execute(
            "INSERT INTO llm_configs (name, provider, api_key, base_url, model_name, report_language, created_at) VALUES (?,?,?,?,?,?,?)",
            (
                cfg.name,
                cfg.provider,
                cfg.api_key,
                cfg.base_url,
                cfg.model_name,
                cfg.report_language,
                now,
            ),
        )
        row_id = cur.lastrowid
        row = conn.execute(
            "SELECT id, name, provider, api_key, base_url, model_name, report_language, created_at FROM llm_configs WHERE id=?",
            (row_id,),
        ).fetchone()
    return dict(row)


@app.put("/api/llm-configs/{config_id}", response_model=LLMConfigItem)
async def update_llm_config(config_id: int, cfg: LLMConfigUpdate):
    """更新指定的 LLM 配置。"""
    with db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM llm_configs WHERE id=?", (config_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="LLM config not found")

        updates = []
        params = []
        if cfg.name is not None:
            updates.append("name=?")
            params.append(cfg.name)
        if cfg.provider is not None:
            updates.append("provider=?")
            params.append(cfg.provider)
        if cfg.api_key is not None:
            updates.append("api_key=?")
            params.append(cfg.api_key)
        if cfg.base_url is not None:
            updates.append("base_url=?")
            params.append(cfg.base_url)
        if cfg.model_name is not None:
            updates.append("model_name=?")
            params.append(cfg.model_name)
        if cfg.report_language is not None:
            updates.append("report_language=?")
            params.append(cfg.report_language)

        if updates:
            params.append(config_id)
            conn.execute(
                f"UPDATE llm_configs SET {', '.join(updates)} WHERE id=?", tuple(params)
            )

        row = conn.execute(
            "SELECT id, name, provider, api_key, base_url, model_name, report_language, created_at FROM llm_configs WHERE id=?",
            (config_id,),
        ).fetchone()
    return dict(row)


@app.delete("/api/llm-configs/{config_id}")
async def delete_llm_config(config_id: int):
    """删除指定 LLM 配置。"""
    with db_conn() as conn:
        conn.execute("DELETE FROM llm_configs WHERE id=?", (config_id,))
    return {"ok": True}


@app.post("/api/report")
async def generate_report(req: ReportRequest):
    """创建后台任务生成研究前置调研报告，并立即返回任务 ID。"""
    api_key = _require_api_key(req.api_key, req.base_url)

    now = datetime.utcnow().isoformat()
    with db_conn() as conn:
        cur = conn.execute(
            "INSERT INTO background_jobs (topic, format_style, status, llm_api_key, llm_model, llm_base_url, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                req.topic,
                req.format_style or "IEEE",
                "running",
                api_key,
                req.model,
                req.base_url or "",
                now,
                now,
            ),
        )
        job_id = cur.lastrowid
        assert job_id is not None

    _run_report_job(
        job_id=job_id,
        topic=req.topic,
        format_style=req.format_style or "IEEE",
        api_key=api_key,
        model=req.model,
        base_url=req.base_url,
        report_language=req.report_language or "zh-CN",
    )
    return {"status": "ok", "job_id": job_id}


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_events(job_id: int):
    """通过 SSE 沟通后台运行的任务进度。"""

    async def event_stream():
        last_event_id = 0
        while True:
            with db_conn() as conn:
                job = conn.execute(
                    "SELECT status, report_id, error FROM background_jobs WHERE id=?",
                    (job_id,),
                ).fetchone()
                if not job:
                    yield f"data: {json.dumps({'status': 'error', 'message': 'Job not found'})}\n\n"
                    break

                events = conn.execute(
                    "SELECT id, status, message, created_at FROM job_events WHERE job_id=? AND id > ? ORDER BY id ASC",
                    (job_id, last_event_id),
                ).fetchall()

            for ev in events:
                last_event_id = ev["id"]
                if ev["status"] == "done":
                    # 解析 message 中的 report_id
                    try:
                        msg_data = json.loads(ev["message"])
                        yield f"data: {json.dumps({'status': 'done', 'report_id': msg_data.get('report_id')})}\n\n"
                    except:
                        yield f"data: {json.dumps({'status': 'done', 'report_id': job['report_id']})}\n\n"
                elif ev["status"] == "error":
                    yield f"data: {json.dumps({'status': 'error', 'message': ev['message']})}\n\n"
                else:
                    yield f"data: {json.dumps({'status': ev['status'], 'message': ev['message']}, ensure_ascii=False)}\n\n"

            if job["status"] in ("done", "error"):
                # 如果任务已经结束且之前没发送对应事件，在这里发送
                if not events:
                    if job["status"] == "done":
                        yield f"data: {json.dumps({'status': 'done', 'report_id': job['report_id']})}\n\n"
                    else:
                        yield f"data: {json.dumps({'status': 'error', 'message': job['error']})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/history", response_model=List[HistoryItem])
async def get_history():
    """返回所有历史报告和运行中的任务列表（按时间倒序）。"""
    with db_conn() as conn:
        # 获取所有 report
        reports = conn.execute(
            "SELECT id, topic, version, file_path, format_style, created_at, status, error FROM reports"
        ).fetchall()

        # 获取所有仍在后台运行或失败但未生成 report 的 job
        jobs = conn.execute(
            "SELECT id, report_id, topic, format_style, created_at, status, error, llm_model, llm_base_url FROM background_jobs WHERE report_id IS NULL"
        ).fetchall()

    items = []
    for r in reports:
        items.append(
            HistoryItem(
                id=r["id"],
                report_id=r["id"],
                topic=r["topic"],
                version=r["version"],
                file_path=r["file_path"],
                format_style=r["format_style"],
                created_at=r["created_at"],
                status=r["status"],
                error=r["error"],
            )
        )
    for j in jobs:
        items.append(
            HistoryItem(
                id=-j["id"],  # 前端需要唯一 id，可以用负数区分
                is_job=True,
                job_id=j["id"],
                topic=j["topic"],
                version="1.0(生成中)",
                file_path="",
                format_style=j["format_style"],
                created_at=j["created_at"],
                status=j["status"],
                error=j["error"],
                llm_model=j["llm_model"],
                llm_base_url=j["llm_base_url"],
            )
        )

    items.sort(key=lambda x: x.created_at, reverse=True)
    return items


@app.get("/api/report/{report_id}", response_model=ReportResponse)
async def get_report(report_id: int):
    """获取单条报告详情。"""
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    report = dict(row)
    return ReportResponse(
        id=report["id"],
        topic=report["topic"],
        version=report["version"],
        file_path=report["file_path"],
        markdown=report["markdown"],
        round1=json.loads(report["round1"] or "{}"),
        round2=json.loads(report["round2"] or "{}"),
        round3=json.loads(report["round3"] or "{}"),
        status=report.get("status", "done"),
        error=report.get("error"),
        created_at=report["created_at"],
    )


@app.get("/api/report/{report_id}/messages", response_model=List[MessageItem])
async def get_messages(report_id: int):
    """获取某报告的对话历史。"""
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE report_id=? ORDER BY id ASC", (report_id,)
        ).fetchall()
    return [MessageItem(**dict(r)) for r in rows]


@app.post("/api/report/{report_id}/chat", response_model=MessageItem)
async def chat(report_id: int, req: ChatRequest):
    """对报告进行对话微调（不重新生成，仅记录对话）。"""
    api_key = _require_api_key(req.api_key, req.base_url)

    # 获取报告
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="报告不存在")

    # 保存用户消息
    now = datetime.utcnow().isoformat()
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO messages (report_id, role, content, created_at) VALUES (?,?,?,?)",
            (report_id, "user", req.message, now),
        )

    # 获取历史消息构建上下文
    with db_conn() as conn:
        prev_msgs = conn.execute(
            "SELECT role, content FROM messages WHERE report_id=? ORDER BY id ASC",
            (report_id,),
        ).fetchall()

    system_prompt = (
        f"你是一位学术助手，帮助用户微调以下研究前置调研报告。\n"
        f"报告主题：{row['topic']}\n\n"
        f"当前报告内容（Markdown）：\n{row['markdown']}\n\n"
        f"请根据用户的要求给出修改建议或直接输出修改后的内容。"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for m in prev_msgs:
        messages.append({"role": m["role"], "content": m["content"]})

    log_path = str(create_log_path(row["topic"]))

    # 调用 LLM
    from .research_prereq import DEFAULT_BASE_URL, DEFAULT_MODEL
    import requests as req_lib

    model = req.model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    base_url = req.base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)

    def _call_llm():
        req_url = f"{base_url.rstrip('/')}/chat/completions"
        append_log(
            log_path,
            "chat_request",
            {
                "method": "POST",
                "url": req_url,
                "messages": messages,
                "model": model,
                "base_url": base_url,
            },
        )
        resp = req_lib.post(
            req_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": messages},
            timeout=120,
        )
        try:
            resp.raise_for_status()
            data = resp.json()
            append_log(log_path, "chat_response", data)
        except Exception as exc:
            append_log(
                log_path,
                "chat_error",
                {
                    "error": str(exc),
                    "status_code": resp.status_code,
                    "text": resp.text[:1000],
                },
            )
            raise ValueError(f"LLM Chat 接口返回了非 JSON: {resp.text[:500]}") from exc
        return data["choices"][0]["message"]["content"]

    try:
        reply = await asyncio.to_thread(_call_llm)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    now2 = datetime.utcnow().isoformat()
    with db_conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages (report_id, role, content, created_at) VALUES (?,?,?,?)",
            (report_id, "assistant", reply, now2),
        )
        msg_id = cur.lastrowid
    assert msg_id is not None

    return MessageItem(
        id=msg_id, report_id=report_id, role="assistant", content=reply, created_at=now2
    )


@app.post("/api/report/{report_id}/regenerate")
async def regenerate_report(report_id: int, req: RegenerateRequest):
    """根据对话指令创建重新生成的后台任务。"""
    api_key = _require_api_key(req.api_key, req.base_url)

    with db_conn() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="报告不存在")

    topic = row["topic"]
    format_style = row["format_style"]
    enhanced_topic = f"{topic}\n\n[微调指令]: {req.instruction}"

    now = datetime.utcnow().isoformat()
    with db_conn() as conn:
        cur = conn.execute(
            "INSERT INTO background_jobs (topic, format_style, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (enhanced_topic, format_style, "running", now, now),
        )
        job_id = cur.lastrowid
        assert job_id is not None

    _run_report_job(
        job_id=job_id,
        topic=enhanced_topic,
        format_style=format_style,
        api_key=api_key,
        model=req.model,
        base_url=req.base_url,
        report_topic=topic,
        post_save_message=f"[重新生成指令]: {req.instruction}",
    )

    return {"status": "ok", "job_id": job_id}


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: int, req: RetryJobRequest):
    """重试一个已经失败的后台任务，使用 job 保存的 LLM 配置。"""
    with db_conn() as conn:
        job = conn.execute(
            "SELECT * FROM background_jobs WHERE id=?", (job_id,)
        ).fetchone()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] not in ("error", "running"):
        raise HTTPException(
            status_code=400, detail="Only error or running jobs can be retried"
        )

    # 优先使用 job 保存的 LLM 配置；请求体中的字段作为后备（兼容旧客户端）
    api_key_to_use = job["llm_api_key"] or req.api_key
    model_to_use = job["llm_model"] or req.model
    base_url_to_use = job["llm_base_url"] or req.base_url
    api_key_to_use = _require_api_key(api_key_to_use, base_url_to_use)

    now = datetime.utcnow().isoformat()
    with db_conn() as conn:
        conn.execute(
            "UPDATE background_jobs SET status='running', error=NULL, updated_at=? WHERE id=?",
            (now, job_id),
        )
        conn.execute("DELETE FROM job_events WHERE job_id=?", (job_id,))

    # Check if this job has a micro-tuning prefix
    topic = job["topic"]
    original_topic = topic
    post_save_message = None
    if "[微调指令]: " in topic:
        parts = topic.split("\n\n[微调指令]: ")
        original_topic = parts[0]
        instruction = parts[1] if len(parts) > 1 else ""
        if instruction:
            post_save_message = f"[重新生成指令]: {instruction}"

    _run_report_job(
        job_id=job_id,
        topic=topic,
        format_style=job["format_style"],
        api_key=api_key_to_use,
        model=model_to_use,
        base_url=base_url_to_use,
        report_topic=original_topic,
        post_save_message=post_save_message,
    )

    return {"status": "ok", "job_id": job_id}


@app.delete("/api/report/{report_id}")
async def delete_report(report_id: int):
    """删除报告记录（不删除 md 文件）。"""
    with db_conn() as conn:
        conn.execute("DELETE FROM messages WHERE report_id=?", (report_id,))
        conn.execute("DELETE FROM reports WHERE id=?", (report_id,))
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
