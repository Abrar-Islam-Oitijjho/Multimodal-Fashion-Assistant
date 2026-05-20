import torch
from transformers import AutoModel, AutoProcessor, Qwen2VLForConditionalGeneration


class ModelLoader:
    def __init__(
        self,
        reasoning_model_name: str,
        retrieval_model_name: str,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.reasoning_model_name = reasoning_model_name
        self.retrieval_model_name = retrieval_model_name
        self.device = device

    def load_reasoning_model(self):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.reasoning_model_name,
            torch_dtype="auto",
        ).to(self.device)

        model.eval()

        processor = AutoProcessor.from_pretrained(
            self.reasoning_model_name,
        )

        return model, processor

    def load_retrieval_model(self):
        model = AutoModel.from_pretrained(
            self.retrieval_model_name,
            torch_dtype="auto",
        ).to(self.device)

        model.eval()

        processor = AutoProcessor.from_pretrained(
            self.retrieval_model_name,
        )

        return model, processor