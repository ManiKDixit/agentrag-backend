from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth.router import router as auth_router
from app.knowledge.router import router as knowledge_router
from app.agents.router import router as agents_router
from app.ml_pipeline.router import router as ml_router

app = FastAPI(
    title="Agentic AI Platform",
    description="RAG, agents, and ML pipelines",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(knowledge_router, prefix="/api/knowledge", tags=["Knowledge Base"])
app.include_router(agents_router, prefix="/api/agents", tags=["Agentic AI"])
app.include_router(ml_router, prefix="/api/ml", tags=["ML Pipeline"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
