from fastapi import Depends, HTTPException, Header
from typing import Optional
from supabase import create_client, Client
from openai import OpenAI

from app.config import get_settings, Settings


def get_supabase_client(
    settings: Settings = Depends(get_settings),
) -> Client:
    return create_client(settings.supabase_url, settings.supabase_key)


def get_supabase_admin(
    settings: Settings = Depends(get_settings),
) -> Client:
    return create_client(settings.supabase_url, settings.supabase_service_key)


def get_openai_client(
    settings: Settings = Depends(get_settings),
) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


# async def get_current_user(
#     authorization: str = Header(...),
#     supabase: Client = Depends(get_supabase_client),
# ) -> dict:
#     try:
#         token = authorization.replace("Bearer ", "")
#         user_response = supabase.auth.get_user(token)
#         if not user_response.user:
#             raise HTTPException(status_code=401, detail="Invalid token")
#         return {"id": user_response.user.id, "email": user_response.user.email}
#     except Exception as e:
#         raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")

async def get_current_user(
    authorization: str = Header(default=None),
    supabase: Client = Depends(get_supabase_client),
) -> dict:
    """
    If a valid token is provided, use that user.
    If no token, fall back to demo user so anyone can try the app.
    """
    if authorization and authorization.startswith("Bearer ") and len(authorization) > 10:
        try:
            token = authorization.replace("Bearer ", "")
            user_response = supabase.auth.get_user(token)
            if user_response.user:
                return {"id": user_response.user.id, "email": user_response.user.email}
        except Exception:
            pass

    # Demo fallback — your real user ID so your documents are visible
    return {"id": "f91a4617-f3b8-4769-9ae4-d5805ecaa152", "email": "manikarnikadixit@gmail.com"}