"""Auth routes for Anzar."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from anzar.config import settings
from anzar.db.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from anzar.db.base import get_db
from anzar.db.models import Settings, User, Workspace

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _workspace_disk_path(user_id: uuid.UUID) -> str:
    """Per-user workspace directory, writable without root.

    Location is overridable via ``ANZAR_WORKSPACES_DIR`` (used by tests) and
    defaults to a user-writable path instead of the historical ``/workspaces``.
    """
    base = os.environ.get("ANZAR_WORKSPACES_DIR") or str(Path.home() / ".anzar" / "workspaces")
    return str(Path(base) / str(user_id))


def get_current_user_dependency(
    authorization: str = Header(default=""),
    db: Session = Depends(get_db),
) -> User:
    token = authorization.replace("Bearer ", "") if authorization else ""
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = get_current_user(db, token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class SocialLoginRequest(BaseModel):
    token: str
    name: str
    email: str
    avatar_url: str | None = None


class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    user: dict


def _get_or_create_social_user(
    db: Session,
    provider: str,
    provider_id: str,
    email: str,
    name: str,
    avatar_url: str | None = None,
) -> User:
    """Find or create user from social login."""
    user = db.query(User).filter(User.auth_provider == provider, User.auth_provider_id == provider_id).first()
    if user:
        return user

    user = db.query(User).filter(User.email == email).first()
    if user:
        user.auth_provider = provider
        user.auth_provider_id = provider_id
        if avatar_url:
            user.avatar_url = avatar_url
        db.commit()
        db.refresh(user)
        return user

    user = User(
        email=email,
        name=name,
        avatar_url=avatar_url,
        auth_provider=provider,
        auth_provider_id=provider_id,
    )
    db.add(user)
    db.flush()

    workspace = Workspace(user_id=user.id, disk_path=_workspace_disk_path(user.id))
    db.add(workspace)
    db.add(Settings(user_id=user.id))

    db.commit()
    db.refresh(user)
    return user


@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=req.email,
        password_hash=hash_password(req.password),
        name=req.name,
        auth_provider="email",
    )
    db.add(user)
    db.flush()

    workspace = Workspace(user_id=user.id, disk_path=_workspace_disk_path(user.id))
    db.add(workspace)
    db.add(Settings(user_id=user.id))

    db.commit()
    db.refresh(user)

    return AuthResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        user={"id": str(user.id), "email": user.email, "name": user.name},
    )


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if user is None or user.password_hash is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return AuthResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        user={"id": str(user.id), "email": user.email, "name": user.name},
    )


@router.post("/refresh", response_model=AuthResponse)
def refresh(req: RefreshRequest, db: Session = Depends(get_db)):
    payload = decode_token(req.refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return AuthResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        user={"id": str(user.id), "email": user.email, "name": user.name},
    )


@router.get("/me")
def get_me(user: User = Depends(get_current_user_dependency), db: Session = Depends(get_db)):
    return {"id": str(user.id), "email": user.email, "name": user.name, "plan": user.plan}


@router.post("/google", response_model=AuthResponse)
def google_login(req: SocialLoginRequest, db: Session = Depends(get_db)):
    """Login with Google OAuth token."""
    if not settings.google_client_id:
        raise HTTPException(status_code=501, detail="Google login not configured")

    try:
        resp = httpx.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={req.token}",
            timeout=10,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid Google token")
    except httpx.HTTPError:
        raise HTTPException(status_code=401, detail="Failed to verify Google token")

    user = _get_or_create_social_user(
        db, provider="google", provider_id=req.token[:32],
        email=req.email, name=req.name, avatar_url=req.avatar_url,
    )

    return AuthResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        user={"id": str(user.id), "email": user.email, "name": user.name},
    )


@router.post("/github", response_model=AuthResponse)
def github_login(req: SocialLoginRequest, db: Session = Depends(get_db)):
    """Login with GitHub OAuth token."""
    try:
        resp = httpx.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {req.token}"},
            timeout=10,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid GitHub token")
        gh_user = resp.json()
    except httpx.HTTPError:
        raise HTTPException(status_code=401, detail="Failed to verify GitHub token")

    user = _get_or_create_social_user(
        db, provider="github", provider_id=str(gh_user.get("id")),
        email=req.email, name=req.name, avatar_url=req.avatar_url,
    )

    return AuthResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        user={"id": str(user.id), "email": user.email, "name": user.name},
    )


@router.get("/config")
def auth_config():
    """Return which OAuth providers are configured (no secrets)."""
    return {
        "google": bool(settings.google_client_id),
        "google_client_id": settings.google_client_id or "",
        "github": bool(settings.github_client_id and settings.github_client_secret),
    }


@router.get("/github/authorize")
def github_authorize():
    """Redirect user to GitHub OAuth consent page."""
    if not settings.github_client_id:
        raise HTTPException(status_code=501, detail="GitHub login not configured")
    params = {
        "client_id": settings.github_client_id,
        "scope": "read:user user:email",
        "allow_signup": "false",
    }
    return RedirectResponse(
        url=f"https://github.com/login/oauth/authorize?{urlencode(params)}"
    )


@router.get("/github/callback")
def github_callback(code: str = Query(default=""), db: Session = Depends(get_db)):
    """Handle GitHub OAuth callback — exchange code for token, create user, redirect to login."""
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")
    if not settings.github_client_id or not settings.github_client_secret:
        raise HTTPException(status_code=501, detail="GitHub login not configured")

    # Exchange code for access token
    try:
        resp = httpx.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
            },
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Failed to exchange GitHub code")
        token_data = resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="No access token from GitHub")
    except httpx.HTTPError:
        raise HTTPException(status_code=401, detail="Failed to connect to GitHub")

    # Get user info
    try:
        user_resp = httpx.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if user_resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Failed to fetch GitHub user info")
        gh_user = user_resp.json()
    except httpx.HTTPError:
        raise HTTPException(status_code=401, detail="Failed to connect to GitHub")

    email = gh_user.get("email") or f"{gh_user.get('login')}@github.local"
    name = gh_user.get("name") or gh_user.get("login") or "GitHub User"
    avatar_url = gh_user.get("avatar_url")

    user = _get_or_create_social_user(
        db, provider="github", provider_id=str(gh_user.get("id")),
        email=email, name=name, avatar_url=avatar_url,
    )

    token = create_access_token(user.id)
    return RedirectResponse(url=f"/login/?token={token}")
