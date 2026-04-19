"""
THE AGENT ORCHESTRATOR — ReAct (Reasoning + Acting) pattern.

1. REASON: LLM thinks about what to do
2. ACT: LLM calls a tool
3. OBSERVE: LLM sees the tool's output
4. Repeat until the LLM has enough info to answer
"""
import json
from typing import AsyncGenerator
from openai import OpenAI

from app.agents.tools import ToolRegistry
from app.agents.memory import ConversationMemory
from app.config import get_settings


# SYSTEM_PROMPT = """You are an intelligent AI assistant with access to a personal knowledge base and other tools.

# KNOWLEDGE BASE CONTENTS:
# {document_context}

# CRITICAL INSTRUCTIONS:
# 1. You KNOW what documents the user has (listed above). When they say "the documents" or "my docs", refer to these.
# 2. When asked about documents or their content, ALWAYS use knowledge_base_search with specific terms from the document titles or topics.
# 3. When comparing documents, search for EACH document separately by using keywords from their titles.
# 4. If the user asks a vague question like "what are the differences between the two docs", you ALREADY KNOW the document titles — search for each one by name.
# 5. Base your answers ONLY on retrieved content — do not fabricate information.
# 6. Format responses clearly with markdown. Use tables for comparisons, bullet points for lists.
# 7. Always cite which document the information came from.

# TOOL USAGE STRATEGY:
# - For questions about uploaded documents → knowledge_base_search (use keywords from document titles)
# - For current events or general knowledge → web_search
# - For calculations → calculator
# - For date/time → get_datetime

# Available tools: {tool_descriptions}
# """


SYSTEM_PROMPT = """You are an AI research assistant that answers questions based on the user's personal knowledge base. You work like Perplexity — grounded answers with inline source citations.

KNOWLEDGE BASE:
{document_context}

SEARCH STRATEGY:
- NEVER search using filenames or document titles as queries. Instead, extract the TOPIC and search with subject keywords.
  Example: Document titled "Fine-Grained_Image_Analysis.pdf" → search for "fine-grained image classification" or "image analysis deep learning"
- When comparing documents, do separate searches for each document's topic
- If the first search returns weak results (low similarity), try different keyword combinations
- Do at least 2-3 searches with varied keywords before concluding information isn't available

RESPONSE FORMAT (Perplexity-style):
- Start with a concise direct answer
- Follow with detailed explanation organized under clear headings
- Cite sources inline using [1], [2] etc. referencing the source chunks
- End with a "Sources" section that lists the FULL document titles, like:
  Sources:
  1. Fine-Grained Image Analysis With Deep Learning: A Survey
  2. Generative AI exists because of the transformer
  NEVER write just numbers or references like "[1]" in the sources list — always write the full document name
- Use markdown formatting: bold for key terms, tables for comparisons, bullet points for lists
- If you genuinely cannot find information in the knowledge base, say so clearly — NEVER fabricate or use your general knowledge as a substitute

CRITICAL RULES:
1. ONLY use information from knowledge_base_search results — do NOT supplement with your own knowledge
2. If retrieval returns nothing useful, say: "I couldn't find specific information about this in your documents."
3. When the user says "the documents" or "my docs", you know their titles from the list above — search by TOPIC, not title

Available tools: {tool_descriptions}
"""

class AgentOrchestrator:
    def __init__(
        self,
        openai_client: OpenAI,
        tool_registry: ToolRegistry,
        memory: ConversationMemory,
        max_iterations: int = 10,
        document_context: str = "No documents available.",
    ):
        self.client = openai_client
        self.tools = tool_registry
        self.memory = memory
        self.settings = get_settings()
        self.max_iterations = max_iterations
        self.document_context = document_context

    async def run(self, user_message: str, conversation_id: str) -> str:
        """Main agent execution loop (non-streaming)."""
        messages = await self.memory.get_messages(conversation_id)

        tool_descriptions = "\n".join(
            f"- {t['function']['name']}: {t['function']['description']}"
            for t in self.tools.get_all_tools_for_llm()
        )
        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                tool_descriptions=tool_descriptions,
                document_context=self.document_context,
            ),
        }

        messages.append({"role": "user", "content": user_message})
        await self.memory.add_message(conversation_id, "user", user_message)

        for iteration in range(self.max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=[system_message] + messages,
                    tools=self.tools.get_all_tools_for_llm(),
                    tool_choice="auto",
                    temperature=self.settings.llm_temperature,
                )

                assistant_message = response.choices[0].message

                if assistant_message.tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_message.tool_calls
                        ],
                    })

                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        print(f"  Tool: {function_name}({arguments})")

                        result = await self.tools.execute_tool(function_name, arguments)
                        print(f"  Result: {result[:200]}...")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        })
                else:
                    final_response = assistant_message.content or "I couldn't generate a response."
                    await self.memory.add_message(
                        conversation_id, "assistant", final_response
                    )
                    return final_response

            except Exception as e:
                print(f"Error in agent loop: {str(e)}")
                return f"An error occurred: {str(e)}"

        return "I reached the maximum number of reasoning steps. Please try a simpler question."
