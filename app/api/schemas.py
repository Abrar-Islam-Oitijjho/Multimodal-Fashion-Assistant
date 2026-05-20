from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """JSON request body for text-only retrieval."""

    user_query: Optional[str] = Field(
        default=None,
        description="Text query, for example: 'black and red plaid dress with Peter Pan collar'.",
    )
    retrieval_top_k: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Number of candidates to retrieve before filtering/reranking.",
    )
    display_top_k: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of final products to return.",
    )
    rerank_top_k: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Number of filtered candidates to rerank. Step 7 clamps this to 20-50 when possible.",
    )
    debug: bool = Field(
        default=False,
        description="Return debug_trace when true.",
    )


class SearchResponse(BaseModel):
    """API response returned by the Step 7 LangGraph workflow."""

    input_type: Optional[str] = None
    user_intent: Optional[str] = None
    retrieval_strategy: Optional[str] = None
    alpha: Optional[float] = None
    search_description: Optional[str] = None
    structured_query: dict[str, Any] = Field(default_factory=dict)
    quality_check: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    retry_reason: Optional[str] = None
    mode: Optional[str] = None
    products: list[dict[str, Any]] = Field(default_factory=list)
    conversation: Optional[list[dict[str, Any]]] = None
    debug_trace: Optional[list[dict[str, Any]]] = None
    error: Optional[str] = None
