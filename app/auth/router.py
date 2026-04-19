# backend/app/auth/router.py
"""
We use Supabase Auth, which handles:
  - Password hashing (bcrypt)
  - JWT token generation & refresh
  - Email verification (optional)
  - OAuth providers (Google, GitHub — can add later)

WHY not roll our own auth?
Auth is security-critical. One mistake (weak hashing, token leaks) = disaster.
Supabase Auth is battle-tested and free.
"""
from fastapi import APIRouter, Depends, HTTPException
from supabase import Client
from pydantic import BaseModel, EmailStr

from app.dependencies import get_supabase_client

router = APIRouter()


class AuthRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    user_id: str


@router.post("/signup", response_model=AuthResponse)
async def signup(
    request: AuthRequest,
    supabase: Client = Depends(get_supabase_client),
):
    """
    Creates a new user account.
    Supabase handles: password hashing, storing credentials, generating JWTs.
    The returned access_token is what the frontend sends with every request.
    """
    try:
        response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password,
        })
        session = response.session
        if not session:
            raise HTTPException(status_code=400, detail="Signup failed — check email for verification")

        return AuthResponse(
            access_token=session.access_token,
            refresh_token=session.refresh_token,
            user_id=response.user.id,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=AuthResponse)
async def login(
    request: AuthRequest,
    supabase: Client = Depends(get_supabase_client),
):
    """
    Authenticates user and returns JWT tokens.

    The access_token expires (default: 1 hour).
    The refresh_token is used to get a new access_token without re-entering password.
    """
    try:
        response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password,
        })
        session = response.session
        return AuthResponse(
            access_token=session.access_token,
            refresh_token=session.refresh_token,
            user_id=response.user.id,
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")


@router.post("/refresh")
async def refresh_token(
    refresh_token: str,
    supabase: Client = Depends(get_supabase_client),
):
    """Uses refresh token to get a new access token without re-login."""
    try:
        response = supabase.auth.refresh_session(refresh_token)
        return {
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Token refresh failed")