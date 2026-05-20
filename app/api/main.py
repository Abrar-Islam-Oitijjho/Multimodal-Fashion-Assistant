from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings


app = FastAPI(
    title="Multimodal Fashion Assistant API",
    description="FastAPI REST API for LangGraph-based fashion retrieval and reasoning.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/images",
    StaticFiles(directory=str(settings.IMAGE_DIR)),
    name="images",
)


app.include_router(router)