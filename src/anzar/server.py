"""FastAPI server for Anzar."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from anzar import __version__
from anzar.api.auth import router as auth_router
from anzar.config import settings
from anzar.db.auth import get_current_user
from anzar.db.base import get_db, init_db
from anzar.db.models import Conversation, Message, Settings, User, Workspace

logger = logging.getLogger("anzar.server")

app = FastAPI(title="Anzar API", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Rate limiting ---
import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMITS = {
    "free": {"requests": 30, "window": 60},
    "pro": {"requests": 120, "window": 60},
    "team": {"requests": 300, "window": 60},
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        token = auth.replace("Bearer ", "") if auth else ""
        if not token:
            return await call_next(request)

        from anzar.db.auth import decode_token
        payload = decode_token(token)
        if not payload:
            return await call_next(request)

        user_id = payload.get("sub", "anon")
        key = f"{user_id}:{request.url.path}"
        now = time.time()
        window = 60

        _rate_store[key] = [t for t in _rate_store[key] if now - t < window]

        # Determine plan limit
        from anzar.db.base import SessionLocal
        db = SessionLocal()
        try:
            from anzar.db.models import User
            u = db.query(User).filter(User.id == user_id).first()
            plan = u.plan if u else "free"
        finally:
            db.close()

        limit = RATE_LIMITS.get(plan, RATE_LIMITS["free"])["requests"]
        if len(_rate_store[key]) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded ({limit} req/min). Upgrade to Pro for higher limits."},
            )

        _rate_store[key].append(now)
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)

app.include_router(auth_router)


async def get_auth_user(
    authorization: str = Header(default=""),
    db: Session = Depends(get_db),
) -> User:
    """Extract and validate JWT from Authorization header."""
    token = authorization.replace("Bearer ", "") if authorization else ""
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = get_current_user(db, token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


class SettingsRequest(BaseModel):
    provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    max_steps: Optional[int] = None


class ProfileRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None


class PasswordRequest(BaseModel):
    current_password: str
    new_password: str


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/api/settings")
def read_settings(user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    settings = db.query(Settings).filter(Settings.user_id == user.id).first()
    if settings is None:
        settings = Settings(user_id=user.id)
        db.add(settings)
        db.commit()
        db.refresh(settings)

    result = {"provider": settings.provider, "model": settings.model, "max_steps": settings.max_steps}
    if settings.api_key_encrypted:
        key = settings.api_key_encrypted
        result["api_key_masked"] = key[:4] + "*" * (len(key) - 8) + key[-4:] if len(key) > 8 else "****"
    return result


@app.put("/api/settings")
def write_settings(req: SettingsRequest, user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    settings = db.query(Settings).filter(Settings.user_id == user.id).first()
    if settings is None:
        settings = Settings(user_id=user.id)
        db.add(settings)

    if req.provider is not None:
        settings.provider = req.provider
    if req.api_key is not None:
        settings.api_key_encrypted = req.api_key
    if req.model is not None:
        settings.model = req.model
    if req.max_steps is not None:
        if not 1 <= req.max_steps <= 500:
            raise HTTPException(status_code=422, detail="max_steps must be between 1 and 500")
        settings.max_steps = req.max_steps

    db.commit()
    db.refresh(settings)
    return {"provider": settings.provider, "model": settings.model, "max_steps": settings.max_steps}


@app.get("/api/profile")
def read_profile(user: User = Depends(get_auth_user)):
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "plan": user.plan,
        "auth_provider": user.auth_provider,
        "avatar_url": user.avatar_url,
        "created_at": user.created_at.isoformat(),
    }


@app.put("/api/profile")
def update_profile(req: ProfileRequest, user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    if req.name is not None:
        user.name = req.name
    if req.email is not None:
        existing = db.query(User).filter(User.email == req.email, User.id != user.id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already in use")
        user.email = req.email
    db.commit()
    db.refresh(user)
    return {"id": str(user.id), "email": user.email, "name": user.name, "plan": user.plan}


@app.put("/api/profile/password")
def update_password(req: PasswordRequest, user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    if user.password_hash is None:
        raise HTTPException(status_code=400, detail="Account uses social login — no password to change")
    from anzar.db.auth import verify_password, hash_password
    if not verify_password(req.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
    user.password_hash = hash_password(req.new_password)
    db.commit()
    return {"status": "ok", "message": "Password updated"}


@app.get("/api/usage/summary")
def usage_summary(user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    from anzar.db.models import Usage
    from sqlalchemy import func as sqlfunc

    total_tokens = db.query(sqlfunc.coalesce(sqlfunc.sum(Usage.tokens_used), 0)).filter(Usage.user_id == user.id).scalar()
    total_actions = db.query(sqlfunc.count(Usage.id)).filter(Usage.user_id == user.id).scalar()
    chat_count = db.query(sqlfunc.count(Usage.id)).filter(Usage.user_id == user.id, Usage.action == "chat").scalar()

    # Usage by day (last 30 days)
    from datetime import timedelta
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    daily_rows = (
        db.query(
            sqlfunc.date(Usage.created_at).label("day"),
            sqlfunc.coalesce(sqlfunc.sum(Usage.tokens_used), 0).label("tokens"),
            sqlfunc.count(Usage.id).label("count"),
        )
        .filter(Usage.user_id == user.id, Usage.created_at >= thirty_days_ago)
        .group_by(sqlfunc.date(Usage.created_at))
        .order_by(sqlfunc.date(Usage.created_at))
        .all()
    )
    daily = [{"date": str(r.day), "tokens": r.tokens, "actions": r.count} for r in daily_rows]

    # Limits per plan
    plan_limits = {"free": 10000, "pro": 100000, "team": 500000}
    limit = plan_limits.get(user.plan, 10000)

    return {
        "total_tokens": total_tokens,
        "total_actions": total_actions,
        "chat_count": chat_count,
        "daily": daily,
        "plan": user.plan,
        "token_limit": limit,
        "token_usage_pct": round((total_tokens / limit) * 100, 1) if limit else 0,
    }


@app.get("/api/usage/history")
def usage_history(user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    from anzar.db.models import Usage
    rows = (
        db.query(Usage)
        .filter(Usage.user_id == user.id)
        .order_by(Usage.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        {"id": str(r.id), "action": r.action, "tokens_used": r.tokens_used, "created_at": r.created_at.isoformat()}
        for r in rows
    ]


@app.get("/api/conversations")
def list_conversations(user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    convs = db.query(Conversation).filter(Conversation.user_id == user.id).order_by(Conversation.created_at.desc()).all()
    return [{"id": str(c.id), "title": c.title, "created_at": c.created_at.isoformat()} for c in convs]


@app.post("/api/conversations")
def new_conversation(title: str = "New Chat", user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    conv = Conversation(user_id=user.id, title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {"id": str(conv.id), "title": conv.title, "created_at": conv.created_at.isoformat()}


@app.get("/api/conversations/{conversation_id}")
def read_conversation(conversation_id: str, user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == uuid.UUID(conversation_id)).first()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = db.query(Message).filter(Message.conversation_id == conv.id).order_by(Message.created_at).all()
    return {
        "conversation": {"id": str(conv.id), "title": conv.title},
        "messages": [{"id": str(m.id), "role": m.role, "content": m.content} for m in messages],
    }


@app.delete("/api/conversations/{conversation_id}")
def remove_conversation(conversation_id: str, user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == uuid.UUID(conversation_id)).first()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conv)
    db.commit()
    return {"status": "deleted"}


@app.post("/api/chat")
def chat(req: ChatRequest, user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    # Get or create conversation
    if req.conversation_id:
        conv = db.query(Conversation).filter(Conversation.id == uuid.UUID(req.conversation_id)).first()
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conv = Conversation(user_id=user.id, title=req.message[:50])
        db.add(conv)
        db.commit()
        db.refresh(conv)

    # Create agent and run
    try:
        from anzar.agent import create_agent
        agent = create_agent(
            db=db,
            user_id=user.id,
            conversation_id=conv.id,
        )
        response = agent.run(req.message)
    except ValueError as e:
        # Agent config error (e.g., missing API key)
        # Fall back to a simple response
        logger.warning("Agent unavailable: %s — using fallback", e)
        response = (
            f"Anzar received: {req.message}\n\n"
            f"Agent is not configured: {e}\n"
            f"Please set your API key in Settings to enable the AI agent."
        )
        # Save fallback response to DB
        assistant_msg = Message(conversation_id=conv.id, role="assistant", content=response)
        db.add(assistant_msg)
        db.commit()
    except Exception as e:
        logger.error("Agent error: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return {"conversation_id": str(conv.id), "response": response}


# ------------------------------------------------------------------
# Workspace / Sandbox endpoints
# ------------------------------------------------------------------


@app.get("/api/workspace/status")
def workspace_status(user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    """Get the current status of the user's workspace container."""
    ws = db.query(Workspace).filter(Workspace.user_id == user.id).first()
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    result = {
        "workspace_id": str(ws.id),
        "status": ws.status,
        "disk_path": ws.disk_path,
        "disk_limit_mb": ws.disk_limit_mb,
        "container_id": ws.container_id[:12] if ws.container_id else None,
        "container_image": ws.container_image,
        "started_at": ws.started_at.isoformat() if ws.started_at else None,
        "stopped_at": ws.stopped_at.isoformat() if ws.stopped_at else None,
        "last_active_at": ws.last_active_at.isoformat() if ws.last_active_at else None,
    }

    # Add live Docker stats if running
    if ws.container_id and ws.status == "running":
        try:
            from anzar.sandbox import get_manager
            manager = get_manager()
            status = manager.container_status(ws.container_id)
            result["docker"] = status
        except Exception:
            pass

    return result


@app.post("/api/workspace/start")
def workspace_start(user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    """Start the user's workspace container."""
    ws = db.query(Workspace).filter(Workspace.user_id == user.id).first()
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    if ws.status == "running" and ws.container_id:
        return {"status": "already_running", "container_id": ws.container_id[:12]}

    try:
        from anzar.sandbox import get_manager
        manager = get_manager()

        if not manager.check_docker():
            # Fallback: just mark as running (subprocess mode)
            ws.status = "running"
            ws.started_at = datetime.now(timezone.utc)
            db.commit()
            return {"status": "running", "mode": "subprocess", "disk_path": ws.disk_path}

        # Create or start container
        info = manager.create_container(
            user_id=str(user.id),
            plan=user.plan,
            disk_path=ws.disk_path,
        )
        manager.start_container(info["container_id"])

        ws.container_id = info["container_id"]
        ws.container_image = info["image"]
        ws.status = "running"
        ws.started_at = datetime.now(timezone.utc)
        ws.stopped_at = None
        ws.last_active_at = datetime.now(timezone.utc)
        db.commit()

        return {
            "status": "running",
            "container_id": info["short_id"],
            "memory_limit": info["memory_limit"],
            "cpu_limit": info["cpu_limit"],
        }

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Failed to start workspace: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to start workspace: {e}")


@app.post("/api/workspace/stop")
def workspace_stop(user: User = Depends(get_auth_user), db: Session = Depends(get_db)):
    """Stop the user's workspace container."""
    ws = db.query(Workspace).filter(Workspace.user_id == user.id).first()
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    if ws.status != "running":
        return {"status": "already_stopped"}

    if ws.container_id:
        try:
            from anzar.sandbox import get_manager
            manager = get_manager()
            if manager.check_docker():
                manager.stop_container(ws.container_id)
        except Exception as e:
            logger.warning("Error stopping container: %s", e)

    ws.status = "stopped"
    ws.stopped_at = datetime.now(timezone.utc)
    db.commit()

    return {"status": "stopped"}


@app.post("/api/workspace/destroy")
def workspace_destroy(
    confirm: str = "",
    user: User = Depends(get_auth_user),
    db: Session = Depends(get_db),
):
    """Destroy the user's container. Files are kept on disk."""
    if confirm != "DESTROY":
        raise HTTPException(
            status_code=400,
            detail="Send confirm='DESTROY' to confirm workspace destruction",
        )

    ws = db.query(Workspace).filter(Workspace.user_id == user.id).first()
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    if ws.container_id:
        try:
            from anzar.sandbox import get_manager
            manager = get_manager()
            if manager.check_docker():
                manager.destroy_container(ws.container_id)
        except Exception as e:
            logger.warning("Error destroying container: %s", e)

    ws.container_id = None
    ws.status = "stopped"
    ws.stopped_at = datetime.now(timezone.utc)
    db.commit()

    return {"status": "destroyed", "disk_path": ws.disk_path, "note": "Files preserved on disk"}


@app.post("/api/workspace/exec")
def workspace_exec(
    req: dict,
    user: User = Depends(get_auth_user),
    db: Session = Depends(get_db),
):
    """Execute a command inside the user's workspace container."""
    command = req.get("command", "")
    if not command:
        raise HTTPException(status_code=400, detail="Command is required")

    ws = db.query(Workspace).filter(Workspace.user_id == user.id).first()
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    if ws.status != "running" and not ws.container_id:
        raise HTTPException(status_code=400, detail="Workspace is not running. Start it first.")

    try:
        from anzar.sandbox import get_manager
        manager = get_manager()

        if ws.container_id and manager.check_docker():
            result = manager.exec_command(ws.container_id, command)
        else:
            result = manager._subprocess_exec(command)

        ws.last_active_at = datetime.now(timezone.utc)
        db.commit()

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}")


@app.get("/api/providers")
def list_providers():
    """Return available LLM providers and their models."""
    from anzar.agent.providers import SUPPORTED_PROVIDERS

    return {
        "providers": [
            {"id": pid, "name": info["name"], "models": info["models"]}
            for pid, info in SUPPORTED_PROVIDERS.items()
        ]
    }


@app.post("/api/workspace/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    user: User = Depends(get_auth_user),
    db: Session = Depends(get_db),
):
    """Upload files to the user's workspace."""
    import re

    ws = db.query(Workspace).filter(Workspace.user_id == user.id).first()
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    workspace_path = Path(ws.disk_path)
    workspace_path.mkdir(parents=True, exist_ok=True)

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    uploaded = []
    errors = []

    for f in files:
        content = await f.read()
        if len(content) > max_bytes:
            errors.append({"filename": f.filename, "error": f"Exceeds {settings.max_upload_size_mb}MB limit"})
            continue

        # Sanitize filename
        safe_name = re.sub(r'[^\w\-.]', '_', f.filename or "unnamed")
        safe_name = safe_name.strip("_.")
        if not safe_name:
            safe_name = "unnamed_file"

        dest = workspace_path / safe_name
        # Auto-rename on collision
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while dest.exists():
                dest = workspace_path / f"{stem}_{counter}{suffix}"
                counter += 1
            safe_name = dest.name

        dest.write_bytes(content)
        size_display = f"{len(content)}B" if len(content) < 1024 else f"{len(content)/1024:.1f}KB"
        uploaded.append({"filename": safe_name, "size_display": size_display, "path": str(dest)})

    return {
        "uploaded": uploaded,
        "errors": errors,
        "total_uploaded": len(uploaded),
    }


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        token = websocket.query_params.get("token", "")
        if not token:
            await websocket.send_json({"error": "Authentication required"})
            await websocket.close()
            return

        from anzar.db.base import SessionLocal
        db = SessionLocal()
        user = get_current_user(db, token)
        if user is None:
            await websocket.send_json({"error": "Invalid token"})
            await websocket.close()
            return

        conversation_id = None
        cached_agent = None
        cached_conv_id = None

        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "chat":
                message = msg.get("message", "")
                conv_id = msg.get("conversation_id")
                model_override = msg.get("model")
                provider_override = msg.get("provider")

                if conv_id:
                    conv = db.query(Conversation).filter(Conversation.id == uuid.UUID(conv_id)).first()
                else:
                    conv = Conversation(user_id=user.id, title=message[:50])
                    db.add(conv)
                    db.commit()
                    db.refresh(conv)

                # Create agent — rebuild if model overrides change or no cache
                from anzar.agent import create_agent
                has_override = bool(model_override or provider_override)
                if has_override or cached_agent is None or cached_conv_id != conv.id:
                    cached_agent = create_agent(
                        db=db,
                        user_id=user.id,
                        conversation_id=conv.id,
                        model_override=model_override,
                        provider_override=provider_override,
                    )
                    cached_conv_id = conv.id

                # Stream tokens
                try:
                    async for token_chunk in cached_agent.astream(message):
                        await websocket.send_json({"type": "token", "content": token_chunk})

                    await websocket.send_json({"type": "done", "conversation_id": str(conv.id)})
                except Exception as e:
                    logger.error("WebSocket agent error: %s", e)
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif msg.get("type") == "quit":
                await websocket.close()
                return

    except WebSocketDisconnect:
        pass
    finally:
        db.close()


STATIC_DIR = Path(__file__).parent / "static"


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import threading
    import uvicorn

    # Start background health check in a separate thread
    def _health_check_loop():
        import time
        while True:
            try:
                from anzar.sandbox import get_manager
                from anzar.db.base import SessionLocal

                db = SessionLocal()
                try:
                    manager = get_manager()
                    if manager.check_docker():
                        results = manager.health_check_all(db)
                        if results:
                            logger.debug("Health check: %d containers checked", len(results))
                except Exception as e:
                    logger.error("Health check error: %s", e)
                finally:
                    db.close()
            except Exception as e:
                logger.error("Health check loop error: %s", e)
            time.sleep(settings.health_check_interval)

    health_thread = threading.Thread(target=_health_check_loop, daemon=True)
    health_thread.start()
    logger.info("Background health check started (interval: %ds)", settings.health_check_interval)

    # Mount frontend static files
    if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
        if (STATIC_DIR / "_next").exists():
            app.mount("/_next", StaticFiles(directory=str(STATIC_DIR / "_next")), name="next")

        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            file_path = STATIC_DIR / full_path
            # Serve exact file if it exists
            if file_path.is_file():
                return FileResponse(str(file_path))
            # Serve index.html inside directory (e.g. /login/ → /login/index.html)
            index_path = file_path / "index.html"
            if index_path.is_file():
                return FileResponse(str(index_path))
            # Fallback to root index.html for SPA client-side routing
            return FileResponse(str(STATIC_DIR / "index.html"))

    print(f"\n  Anzar v{__version__} — AI Software Engineer Agent")
    print(f"  Server running at http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")

    uvicorn.run(app, host=host, port=port, log_level="info")
