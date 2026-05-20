from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from PIL import Image


class RetrievalService:
    def __init__(
        self,
        embedding_generator,
        image_index_path: str | Path,
        text_index_path: str | Path,
        image_dir: str | Path,
    ):
        self.embedding_generator = embedding_generator

        self.image_index_path = Path(image_index_path)
        self.text_index_path = Path(text_index_path)
        self.image_dir = Path(image_dir)

        self.image_index = faiss.read_index(str(self.image_index_path))
        self.text_index = faiss.read_index(str(self.text_index_path))

    def search_by_image(self, image, top_k: int = 10):
        image_embedding = self.embedding_generator.embed_image(image)
        distances, indices = self.image_index.search(image_embedding, top_k)

        return distances[0], indices[0]

    def search_by_text(self, text: str, top_k: int = 10):
        text_embedding = self.embedding_generator.embed_text(text)
        distances, indices = self.text_index.search(text_embedding, top_k)

        return distances[0], indices[0]

    def combine_scores(
        self,
        image_indices=None,
        image_scores=None,
        text_indices=None,
        text_scores=None,
        alpha: float = 0.4,
    ) -> dict[int, float]:
        image_dict = {}
        text_dict = {}

        if image_indices is not None and image_scores is not None:
            image_dict = {
                int(idx): float(score)
                for idx, score in zip(image_indices, image_scores)
                if idx != -1
            }

        if text_indices is not None and text_scores is not None:
            text_dict = {
                int(idx): float(score)
                for idx, score in zip(text_indices, text_scores)
                if idx != -1
            }

        all_candidates = set(image_dict.keys()).union(set(text_dict.keys()))

        combined_scores = {}

        for idx in all_candidates:
            image_score = image_dict.get(idx, 0.0)
            text_score = text_dict.get(idx, 0.0)

            if image_dict and text_dict:
                final_score = alpha * image_score + (1 - alpha) * text_score
            elif image_dict:
                final_score = image_score
            else:
                final_score = text_score

            combined_scores[idx] = final_score

        return dict(
            sorted(
                combined_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def retrieve(
        self,
        query_text: str | None,
        query_image=None,
        top_k: int = 10,
        alpha: float = 0.4,
    ) -> dict[int, float]:
        image_indices = image_scores = None
        text_indices = text_scores = None

        if query_image is not None:
            image_scores, image_indices = self.search_by_image(
                query_image,
                top_k=top_k,
            )

        if query_text:
            text_scores, text_indices = self.search_by_text(
                query_text,
                top_k=top_k,
            )

        combined_scores = self.combine_scores(
            image_indices=image_indices,
            image_scores=image_scores,
            text_indices=text_indices,
            text_scores=text_scores,
            alpha=alpha,
        )

        return combined_scores

    def get_top_products(
        self,
        combined_scores: dict[int, float],
        master_df: pd.DataFrame,
        top_k: int = 6,
    ) -> list[dict]:
        top_indices = list(combined_scores.keys())[:top_k]

        products = []

        for idx in top_indices:
            row = master_df.iloc[idx]

            image_id = str(row["image_id"])
            item_id = str(row["item_id"])

            image_path_jpg = self.image_dir / f"{image_id}_{item_id}.jpg"
            image_path_png = self.image_dir / f"{image_id}_{item_id}.png"

            if image_path_jpg.exists():
                image_path = image_path_jpg
            elif image_path_png.exists():
                image_path = image_path_png
            else:
                image_path = None

            products.append(
                {
                    "index": int(idx),
                    "score": float(combined_scores[idx]),
                    "image_id": image_id,
                    "item_id": item_id,
                    "category_name": row.get("category_name", None),
                    "description": row.get("description", None),
                    "image_path": str(image_path) if image_path else None,
                }
            )

        return products

    def load_product_images(self, products: list[dict]) -> list[Image.Image]:
        images = []

        for product in products:
            image_path = product.get("image_path")

            if image_path:
                image = Image.open(image_path).convert("RGB")
                images.append(image)

        return images