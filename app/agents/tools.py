# backend/app/agents/tools.py
"""
TOOLS: The capabilities available to the agent.

KEY CONCEPT — What makes AI "agentic"?
=======================================
A regular LLM can only generate text. An AGENT can:
1. Decide WHICH tool to use based on the question
2. Format the correct INPUT for that tool
3. EXECUTE the tool
4. INTERPRET the results
5. Decide if it needs MORE tools or can answer now

Each tool is defined with:
- name: unique identifier the LLM uses to call it
- description: tells the LLM WHEN to use this tool (this is crucial!)
- parameters: JSON schema defining expected inputs
- function: the actual Python code that runs
"""
import httpx
import json
from typing import Any, Callable
from dataclasses import dataclass, field


@dataclass
class Tool:
    """Represents a single tool available to the agent."""
    name: str
    description: str
    parameters: dict                    # JSON Schema for the tool's input
    function: Callable                  # The actual function to execute
    requires_confirmation: bool = False # Some tools need user approval first


class ToolRegistry:
    """
    Central registry of all available tools.

    WHY a registry?
    1. Easy to add/remove tools dynamically
    2. The agent gets the full tool list as part of its prompt
    3. Clean separation: tool logic lives here, agent logic lives elsewhere
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def get_all_tools_for_llm(self) -> list[dict]:
        """
        Formats all tools into the OpenAI function-calling schema.
        This is sent to the LLM so it knows what tools exist.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute_tool(self, name: str, arguments: dict) -> str:
        """Executes a tool and returns the result as a string."""
        tool = self.get_tool(name)
        try:
            result = await tool.function(**arguments)
            return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        except Exception as e:
            return f"Tool execution error: {str(e)}"


# ==============================
# TOOL IMPLEMENTATIONS
# ==============================

async def web_search_tool(query: str) -> dict:
    """
    Searches the web using a search API.
    In production, use Google Custom Search, Serper, or Tavily.
    """
    # Using a free search API for demo purposes
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json"},
        )
        data = response.json()
        return {
            "query": query,
            "results": data.get("AbstractText", "No results found"),
            "source": data.get("AbstractURL", ""),
        }


async def calculator_tool(expression: str) -> dict:
    """
    Evaluates mathematical expressions safely.
    WHY not just eval()? eval() is a MASSIVE security hole —
    it can execute arbitrary Python code. We restrict to math operations.
    """
    import ast
    import operator

    # Safe operators only
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def safe_eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
        else:
            raise ValueError(f"Unsupported operation")

    tree = ast.parse(expression, mode="eval")
    result = safe_eval(tree.body)
    return {"expression": expression, "result": result}


async def datetime_tool() -> dict:
    """Returns current date and time. Agents often need this for temporal reasoning."""
    from datetime import datetime
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
    }


def create_default_registry(retrieval_service=None, user_id=None) -> ToolRegistry:
    """
    Creates a ToolRegistry with all default tools.
    The RAG tool is special — it's created dynamically with the user's context.
    """
    registry = ToolRegistry()

    registry.register(Tool(
        name="web_search",
        description="Search the web for current information. Use this when the question is about recent events, public knowledge, or when the knowledge base doesn't have the answer.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
        function=web_search_tool,
    ))

    registry.register(Tool(
        name="calculator",
        description="Evaluate mathematical expressions. Use for any calculations.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '2 + 3 * 4'"},
            },
            "required": ["expression"],
        },
        function=calculator_tool,
    ))

    registry.register(Tool(
        name="get_datetime",
        description="Get the current date and time.",
        parameters={"type": "object", "properties": {}},
        function=datetime_tool,
    ))

    # The RAG tool — this is what makes the knowledge base queryable
    if retrieval_service and user_id:
        async def rag_query_tool(query: str) -> dict:
            chunks = await retrieval_service.retrieve(query, user_id, top_k=5)
            if not chunks:
                return {"answer": "No relevant information found in knowledge base."}
            return {
                "context": [
                    {"content": c.content, "similarity": c.similarity}
                    for c in chunks
                ],
            }

        registry.register(Tool(
            name="knowledge_base_search",
            description="Search the user's personal knowledge base for information from their uploaded documents. ALWAYS try this first before web search for domain-specific questions.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
            function=rag_query_tool,
        ))

    return registry