import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image


class DataPreprocessor:
    def load_annotations(self, file_path: str | Path) -> list[dict]:
        file_path = Path(file_path)
        rows = []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_id = file_path.name[:6]
        pair_id = data.get("pair_id")
        source = data.get("source")

        for key, item in data.items():
            if not key.startswith("item"):
                continue

            row = {
                "image_id": image_id,
                "pair_id": pair_id,
                "source": source,
                "item_id": key,
                "category_id": item.get("category_id"),
                "category_name": item.get("category_name"),
                "style": item.get("style"),
                "bbox": item.get("bounding_box"),
                "landmarks": item.get("landmarks"),
                "segmentation": item.get("segmentation"),
                "scale": item.get("scale"),
                "occlusion": item.get("occlusion"),
                "zoom_in": item.get("zoom_in"),
                "viewpoint": item.get("viewpoint"),
            }

            rows.append(row)

        return rows

    def load_all_annotations(self, annotation_dir: str | Path) -> pd.DataFrame:
        annotation_dir = Path(annotation_dir)
        all_rows = []

        for json_file in annotation_dir.glob("*.json"):
            rows = self.load_annotations(json_file)
            all_rows.extend(rows)

        return pd.DataFrame(all_rows)

    def crop_images(
        self,
        image_dir: str | Path,
        df: pd.DataFrame,
        save_dir: str | Path,
    ) -> None:
        image_dir = Path(image_dir)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for _, row in df.iterrows():
            image_id = row["image_id"]
            item_id = row["item_id"]
            bbox = row["bbox"]

            if bbox is None:
                continue

            image_path = image_dir / f"{image_id}.jpg"

            if not image_path.exists():
                continue

            x1, y1, x2, y2 = bbox

            image = Image.open(image_path).convert("RGB")
            cropped_image = image.crop((x1, y1, x2, y2))

            save_path = save_dir / f"{image_id}_{item_id}.jpg"
            cropped_image.save(save_path)

    def get_all_images_and_mapping(
        self,
        image_dir: str | Path,
        df: pd.DataFrame,
    ) -> Tuple[List[Image.Image], list[tuple[str, str]]]:
        image_dir = Path(image_dir)

        images = []
        ids = []

        for _, row in df.iterrows():
            image_id = str(row["image_id"])
            item_id = str(row["item_id"])

            jpg_path = image_dir / f"{image_id}_{item_id}.jpg"
            png_path = image_dir / f"{image_id}_{item_id}.png"

            if jpg_path.exists():
                image_path = jpg_path
            elif png_path.exists():
                image_path = png_path
            else:
                continue

            image = Image.open(image_path).convert("RGB")
            images.append(image)
            ids.append((image_id, item_id))

        return images, ids