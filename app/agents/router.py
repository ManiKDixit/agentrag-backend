from fastapi import APIRouter, Depends
from pydantic import BaseModel
from supabase import Client
from openai import OpenAI

from app.dependencies import get_supabase_admin, get_openai_client, get_current_user
from app.agents.orchestrator import AgentOrchestrator
from app.agents.tools import create_default_registry
from app.agents.memory import ConversationMemory
from app.knowledge.retrieval import RetrievalService

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


async def get_document_context(supabase: Client, user_id: str) -> str:
    """Fetches the list of documents so the agent knows what's in the knowledge base."""
    response = (
        supabase.table("documents")
        .select("title, source_type, metadata, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    if not response.data:
        return "The user's knowledge base is empty — no documents uploaded yet."

    doc_list = []
    for i, doc in enumerate(response.data, 1):
        chunks = doc.get("metadata", {}).get("chunk_count", "unknown")
        doc_list.append(f'  {i}. "{doc["title"]}" ({doc["source_type"]}, {chunks} chunks)')

    return (
        f"The user has {len(response.data)} document(s) in their knowledge base:\n"
        + "\n".join(doc_list)
    )


@router.post("/chat")
async def chat(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin),
    openai_client: OpenAI = Depends(get_openai_client),
):
    """Main chat endpoint — non-streaming, reliable."""
    memory = ConversationMemory(supabase)

    if request.conversation_id:
        conversation_id = request.conversation_id
    else:
        conversation_id = await memory.create_conversation(user["id"])

    retrieval_service = RetrievalService(supabase, openai_client)
    tool_registry = create_default_registry(retrieval_service, user["id"])
    doc_context = await get_document_context(supabase, user["id"])

    agent = AgentOrchestrator(
        openai_client, tool_registry, memory, document_context=doc_context
    )
    response = await agent.run(request.message, conversation_id)

    return {
        "conversation_id": conversation_id,
        "response": response,
    }


@router.get("/conversations")
async def list_conversations(
    user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin),
):
    response = (
        supabase.table("conversations")
        .select("*")
        .eq("user_id", user["id"])
        .order("updated_at", desc=True)
        .execute()
    )
    return response.data


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    user: dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_admin),
):
    """Retrieve messages for a specific conversation."""
    # Verify conversation belongs to user
    conv = (
        supabase.table("conversations")
        .select("id")
        .eq("id", conversation_id)
        .eq("user_id", user["id"])
        .execute()
    )
    if not conv.data:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Conversation not found")

    response = (
        supabase.table("messages")
        .select("role, content, created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .execute()
    )
    return response.data
