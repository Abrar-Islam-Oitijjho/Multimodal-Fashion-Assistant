import pandas as pd
import torch

from app.config import settings
from app.model_loader import ModelLoader
from app.embeddings import EmbeddingGenerator
from app.reasoning import ReasoningService
from app.retrieval import RetrievalService
from app.chatbot import FashionChatbot


class AppServices:
    def __init__(self):
        self.master_df = None

        self.reasoning_model = None
        self.reasoning_processor = None

        self.retrieval_model = None
        self.retrieval_processor = None

        self.embedding_generator = None
        self.reasoning_service = None
        self.retrieval_service = None
        self.chatbot = None

    def load_master_df(self):
        self.master_df = pd.read_csv(
            settings.NEW_MASTER_CSV_PATH,
            dtype={"image_id": str},
        )

        return self.master_df

    def load_models(self):
        loader = ModelLoader(
            reasoning_model_name=settings.REASONING_MODEL_NAME,
            retrieval_model_name=settings.RETRIEVAL_MODEL_NAME,
            device=settings.DEVICE,
        )

        self.reasoning_model, self.reasoning_processor = loader.load_reasoning_model()
        self.retrieval_model, self.retrieval_processor = loader.load_retrieval_model()

    def build_services(self):
        self.embedding_generator = EmbeddingGenerator(
            model=self.retrieval_model,
            processor=self.retrieval_processor,
        )

        self.reasoning_service = ReasoningService(
            model=self.reasoning_model,
            processor=self.reasoning_processor,
        )

        self.retrieval_service = RetrievalService(
            embedding_generator=self.embedding_generator,
            image_index_path=settings.IMAGE_FAISS_INDEX_PATH,
            text_index_path=settings.TEXT_FAISS_INDEX_PATH,
            image_dir=settings.CROPPED_IMAGE_DIR,
        )

        self.chatbot = FashionChatbot(
            reasoning_service=self.reasoning_service,
            retrieval_service=self.retrieval_service,
            master_df=self.master_df,
        )

    def load_all(self):
        self.load_master_df()
        self.load_models()
        self.build_services()

        return self.chatbot

    def clear_gpu_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app_services = AppServices()


def get_chatbot():
    if app_services.chatbot is None:
        app_services.load_all()

    return app_services.chatbot