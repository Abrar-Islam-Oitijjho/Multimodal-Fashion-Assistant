from pathlib import Path


class Settings:
    BASE_DIR = Path(__file__).resolve().parents[1]

    DATA_DIR = BASE_DIR / "Data"
    TRAIN_DIR = DATA_DIR / "train"

    ANNO_DIR = DATA_DIR / "annos"
    IMAGE_DIR = TRAIN_DIR / "image"
    CROPPED_IMAGE_DIR = DATA_DIR / "cropped_image_unique"

    MASTER_CSV_PATH = DATA_DIR / "master_csv.csv"
    NEW_MASTER_CSV_PATH = DATA_DIR / "new_master_csv.csv"

    FAISS_INDEX_DIR = DATA_DIR
    IMAGE_FAISS_INDEX_PATH = FAISS_INDEX_DIR / "faiss_image_siglip2_base.index"
    TEXT_FAISS_INDEX_PATH = FAISS_INDEX_DIR / "faiss_text_siglip2_base.index"

    REASONING_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
    RETRIEVAL_MODEL_NAME = "google/siglip2-base-patch16-224"

    DEVICE = "cuda"


settings = Settings()