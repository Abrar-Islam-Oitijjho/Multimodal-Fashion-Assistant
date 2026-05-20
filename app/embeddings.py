from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class EmbeddingGenerator:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def _extract_embedding_tensor(self, outputs):
        """
        Handles different Hugging Face output formats.
        Some models return a tensor directly.
        Some return BaseModelOutputWithPooling.
        Some return objects with pooler_output or image_embeds/text_embeds.
        """

        if isinstance(outputs, torch.Tensor):
            return outputs

        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds

        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            return outputs.text_embeds

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state[:, 0, :]

        if isinstance(outputs, tuple):
            for item in outputs:
                if isinstance(item, torch.Tensor):
                    return item

        raise ValueError("Could not extract embedding tensor from model output.")

    def _normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings / embeddings.norm(
            p=2,
            dim=-1,
            keepdim=True,
        )

    def embed_image(self, images) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                outputs = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)

        image_embeddings = self._extract_embedding_tensor(outputs)
        image_embeddings = self._normalize(image_embeddings)

        return image_embeddings.cpu().numpy().astype(np.float32)

    def embed_text(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                outputs = self.model.get_text_features(**inputs)
            else:
                outputs = self.model(**inputs)

        text_embeddings = self._extract_embedding_tensor(outputs)
        text_embeddings = self._normalize(text_embeddings)

        return text_embeddings.cpu().numpy().astype(np.float32)

    def build_image_faiss_index(
        self,
        images: Sequence[Image.Image],
        save_path: str | Path,
        batch_size: int = 4,
    ) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        index = None
        num_batches = (len(images) + batch_size - 1) // batch_size

        for start in tqdm(range(0, len(images), batch_size), total=num_batches):
            batch_images = images[start:start + batch_size]
            image_embeddings = self.embed_image(batch_images)

            if index is None:
                dim = image_embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)

            index.add(image_embeddings)

            del image_embeddings

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        faiss.write_index(index, str(save_path))

    def build_text_faiss_index(
        self,
        texts,
        save_path: str | Path,
        batch_size: int = 4,
    ) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        texts = list(texts)

        index = None
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for start in tqdm(range(0, len(texts), batch_size), total=num_batches):
            batch_texts = texts[start:start + batch_size]
            text_embeddings = self.embed_text(batch_texts)

            if index is None:
                dim = text_embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)

            index.add(text_embeddings)

            del text_embeddings

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        faiss.write_index(index, str(save_path))