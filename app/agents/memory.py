# backend/app/agents/memory.py
"""
CONVERSATION MEMORY
====================
Agents need memory to have coherent multi-turn conversations.

Without memory:
  User: "What's the capital of France?"
  Agent: "Paris."
  User: "What's its population?"
  Agent: "What are you referring to?" ← FAIL

With memory:
  User: "What's the capital of France?"
  Agent: "Paris."
  User: "What's its population?"
  Agent: "Paris has a population of about 2.1 million." ← understands "its" = Paris

We store messages in Supabase and load the last N messages as context.
"""
from typing import List
from supabase import Client


class ConversationMemory:
    def __init__(self, supabase: Client, max_messages: int = 20):
        """
        max_messages: how many past messages to include as context.

        WHY limit? Each message consumes tokens in the LLM's context window.
        20 messages ≈ 2000-4000 tokens, leaving room for system prompt + tools + answer.
        For longer conversations, you'd implement summarisation of older messages.
        """
        self.supabase = supabase
        self.max_messages = max_messages

    async def get_messages(self, conversation_id: str) -> List[dict]:
        """Retrieves recent messages for a conversation."""
        response = (
            self.supabase.table("messages")
            .select("role, content")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)  # chronological
            .limit(self.max_messages)
            .execute()
        )
        return [{"role": m["role"], "content": m["content"]} for m in response.data]

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tool_calls: dict = None,
        tool_results: dict = None,
    ):
        """Persists a message to the database."""
        self.supabase.table("messages").insert({
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }).execute()

    async def create_conversation(self, user_id: str, title: str = "New Conversation") -> str:
        """Creates a new conversation and returns its ID."""
        response = self.supabase.table("conversations").insert({
            "user_id": user_id,
            "title": title,
        }).execute()
        return response.data[0]["id"]