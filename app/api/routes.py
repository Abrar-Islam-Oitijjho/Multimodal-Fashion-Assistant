from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.api.schemas import SearchRequest, SearchResponse
from app.graph_workflow_step7 import fashion_graph


router = APIRouter()


def _to_json_safe(value: Any) -> Any:
    """Convert common ML/Pandas/NumPy/PIL objects into JSON-safe values."""
    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Image.Image):
        # Do not return raw PIL image objects through JSON.
        return None

    if isinstance(value, dict):
        return {str(key): _to_json_safe(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]

    # NumPy scalar support without making NumPy a hard dependency here.
    if hasattr(value, "item"):
        try:
            return _to_json_safe(value.item())
        except Exception:
            pass

    # Pandas NaN/NA fallback.
    try:
        if value != value:  # noqa: PLR0124
            return None
    except Exception:
        pass

    return str(value)


def _build_response(result: dict[str, Any], debug: bool) -> dict[str, Any]:
    """Select the fields that should be exposed by the API."""
    response = {
        "input_type": result.get("input_type"),
        "user_intent": result.get("user_intent"),
        "retrieval_strategy": result.get("retrieval_strategy"),
        "alpha": result.get("alpha"),
        "search_description": result.get("search_description"),
        "structured_query": result.get("structured_query", {}),
        "quality_check": result.get("quality_check", {}),
        "retry_count": result.get("retry_count", 0),
        "retry_reason": result.get("retry_reason"),
        "mode": result.get("mode"),
        "products": add_image_urls(result.get("products", [])),
        "conversation": result.get("conversation"),
        "error": result.get("error"),
    }

    if debug:
        response["debug_trace"] = result.get("debug_trace", [])

    return _to_json_safe(response)


def add_image_urls(products: list[dict]) -> list[dict]:
    updated_products = []

    for product in products:
        product = dict(product)

        image_path = product.get("image_path")

        if image_path:
            image_name = Path(image_path).name
            product["image_url"] = f"/images/{image_name}"
        else:
            product["image_url"] = None

        updated_products.append(product)

    return updated_products


def _invoke_step7_graph(
    *,
    user_query: Optional[str],
    user_image: Optional[Image.Image],
    retrieval_top_k: int,
    display_top_k: int,
    rerank_top_k: int,
) -> dict[str, Any]:
    """Run the frozen Step 7 LangGraph workflow."""
    return fashion_graph.invoke(
        {
            "user_query": user_query,
            "user_image": user_image,
            "retrieval_top_k": retrieval_top_k,
            "display_top_k": display_top_k,
            "rerank_top_k": rerank_top_k,
            "retry_count": 0,
        }
    )


@router.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Step 7 Fashion Assistant API is running.",
    }


@router.post("/search", response_model=SearchResponse)
def search_text(request: SearchRequest) -> dict[str, Any]:
    """Text-only product search endpoint."""
    try:
        result = _invoke_step7_graph(
            user_query=request.user_query,
            user_image=None,
            retrieval_top_k=request.retrieval_top_k,
            display_top_k=request.display_top_k,
            rerank_top_k=request.rerank_top_k,
        )
        return _build_response(result, debug=request.debug)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc


@router.post("/search/image", response_model=SearchResponse)
async def search_with_image(
    user_query: Optional[str] = Form(default=None),
    image: UploadFile = File(...),
    retrieval_top_k: int = Form(default=30),
    display_top_k: int = Form(default=6),
    rerank_top_k: int = Form(default=50),
    debug: bool = Form(default=False),
) -> dict[str, Any]:
    """Image or image+text product search endpoint."""
    if retrieval_top_k < 1 or retrieval_top_k > 100:
        raise HTTPException(status_code=422, detail="retrieval_top_k must be between 1 and 100.")
    if display_top_k < 1 or display_top_k > 20:
        raise HTTPException(status_code=422, detail="display_top_k must be between 1 and 20.")
    if rerank_top_k < 1 or rerank_top_k > 100:
        raise HTTPException(status_code=422, detail="rerank_top_k must be between 1 and 100.")

    try:
        raw_bytes = await image.read()
        pil_image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc

    try:
        result = _invoke_step7_graph(
            user_query=user_query,
            user_image=pil_image,
            retrieval_top_k=retrieval_top_k,
            display_top_k=display_top_k,
            rerank_top_k=rerank_top_k,
        )
        return _build_response(result, debug=debug)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image search failed: {exc}") from exc
