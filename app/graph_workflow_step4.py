import re
from typing import Any, Optional, TypedDict

from PIL import Image
from langgraph.graph import END, START, StateGraph

from app.services import get_chatbot


class FashionState(TypedDict, total=False):
    user_query: Optional[str]
    user_image: Optional[Image.Image]

    retrieval_top_k: int
    display_top_k: int
    alpha: float

    # Step 1 additions
    has_text: bool
    has_image: bool
    input_type: str
    user_intent: str
    retrieval_strategy: str
    debug_trace: list[dict]

    # Step 3 addition
    structured_query: dict[str, Any]

    # Step 4 additions
    image_candidates: list[dict]
    text_candidates: list[dict]
    merged_candidates: list[dict]

    search_description: str
    products: list[dict]
    product_images: list[Any]
    conversation: list[dict]
    mode: str
    error: Optional[str]


def _add_trace(state: FashionState, node: str, values: dict) -> list[dict]:
    trace = list(state.get("debug_trace", []))
    trace.append(
        {
            "node": node,
            **values,
        }
    )
    return trace


def _is_vague_text(text: str) -> bool:
    text = (text or "").lower().strip()

    vague_phrases = [
        "find similar",
        "similar",
        "same",
        "like this",
        "like that",
        "this one",
        "that one",
        "something like this",
        "something like that",
        "match this",
        "find this",
        "show similar",
        "show me similar",
    ]

    if not text:
        return True

    return any(phrase in text for phrase in vague_phrases)


def _find_matches(text: str, vocabulary: list[str]) -> list[str]:
    """Return vocabulary terms that appear as whole phrases in text."""
    matches = []
    for term in vocabulary:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text):
            matches.append(term)
    return matches


def _first_match(text: str, vocabulary: list[str]) -> Optional[str]:
    matches = _find_matches(text, vocabulary)
    return matches[0] if matches else None


def _extract_structured_query(text: str) -> dict[str, Any]:
    """
    Lightweight rule-based product attribute extraction.
    """
    normalized_text = (text or "").lower().strip()

    product_types = [
        "t-shirt",
        "shirt",
        "top",
        "dress",
        "jacket",
        "jeans",
        "pants",
        "trousers",
        "skirt",
        "shorts",
        "hoodie",
        "sweater",
        "coat",
        "blazer",
        "kurti",
        "saree",
        "shoes",
        "sneakers",
        "sandals",
        "boots",
        "bag",
    ]

    colors = [
        "black",
        "white",
        "red",
        "blue",
        "green",
        "yellow",
        "pink",
        "purple",
        "orange",
        "brown",
        "grey",
        "gray",
        "beige",
        "cream",
        "navy",
        "maroon",
        "olive",
        "gold",
        "silver",
    ]

    patterns = [
        "plaid",
        "checkered",
        "checked",
        "striped",
        "stripe",
        "floral",
        "printed",
        "solid",
        "polka dot",
        "polka-dot",
        "animal print",
        "geometric",
        "graphic",
    ]

    collars = [
        "peter pan collar",
        "shirt collar",
        "mandarin collar",
        "crew neck",
        "v-neck",
        "round neck",
        "collared",
        "hooded",
        "turtleneck",
    ]

    sleeves = [
        "sleeveless",
        "short sleeve",
        "short sleeves",
        "long sleeve",
        "long sleeves",
        "half sleeve",
        "half sleeves",
        "three quarter sleeve",
        "3/4 sleeve",
        "cap sleeve",
    ]

    materials = [
        "cotton",
        "denim",
        "leather",
        "silk",
        "linen",
        "wool",
        "polyester",
        "chiffon",
        "knit",
    ]

    fits = [
        "slim fit",
        "regular fit",
        "relaxed fit",
        "oversized",
        "loose fit",
        "skinny",
        "straight fit",
    ]

    styles = [
        "casual",
        "formal",
        "party",
        "sporty",
        "ethnic",
        "western",
        "office",
        "streetwear",
        "vintage",
        "summer",
        "winter",
    ]

    constraints = []

    constraint_patterns = [
        r"\bwithout\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
        r"\bnot\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
        r"\bno\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
        r"\bexcluding\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
        r"\bexclude\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
        r"\bavoid\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
        r"\bremove\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
        r"\bexcept\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
        r"\bother than\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
        r"\bdoes not have\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
        r"\bdoesn't have\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
        r"\bshould not have\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
        r"\bwith no\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
    ]

    for pattern in constraint_patterns:
        for match in re.finditer(pattern, normalized_text):
            value = match.group(1).strip()
            if value:
                constraints.append(value)

    structured_query = {
        "raw_text": text or "",
        "product_type": _first_match(normalized_text, product_types),
        "colors": _find_matches(normalized_text, colors),
        "pattern": _first_match(normalized_text, patterns),
        "collar": _first_match(normalized_text, collars),
        "sleeve": _first_match(normalized_text, sleeves),
        "material": _first_match(normalized_text, materials),
        "fit": _first_match(normalized_text, fits),
        "style": _first_match(normalized_text, styles),
        "constraints": constraints,
    }

    return structured_query


def _safe_value(value: Any) -> Any:
    """Convert pandas NaN/None into clean Python None for API/notebook outputs."""
    if value is None:
        return None

    try:
        if value != value:  # catches NaN without importing pandas here
            return None
    except Exception:
        pass

    return value


def _build_image_path(retrieval_service, image_id: str, item_id: str) -> Optional[str]:
    image_path_jpg = retrieval_service.image_dir / f"{image_id}_{item_id}.jpg"
    image_path_png = retrieval_service.image_dir / f"{image_id}_{item_id}.png"

    if image_path_jpg.exists():
        return str(image_path_jpg)

    if image_path_png.exists():
        return str(image_path_png)

    return None


def _candidate_from_index(
    idx: int,
    score: float,
    score_name: str,
    master_df,
    retrieval_service,
) -> dict:
    """
    Convert one FAISS result index into a product candidate.

    Important: FAISS gives row indices, but Step 4 merging should use the real
    product identity: image_id + item_id.
    """
    row = master_df.iloc[int(idx)]

    image_id = str(row["image_id"])
    item_id = str(row["item_id"])
    product_key = f"{image_id}__{item_id}"

    candidate = {
        "product_key": product_key,
        "index": int(idx),
        "image_id": image_id,
        "item_id": item_id,
        "category_name": _safe_value(row.get("category_name", None)),
        "description": _safe_value(row.get("description", None)),
        "attributes": _safe_value(row.get("attributes", None)),
        "image_path": _build_image_path(retrieval_service, image_id, item_id),
        "image_score": None,
        "text_score": None,
        "combined_score": None,
    }

    candidate[score_name] = float(score)
    return candidate


def _deduplicate_candidates(candidates: list[dict], score_name: str) -> list[dict]:
    """Keep the best candidate for each product key inside one retrieval path."""
    best_by_key = {}

    for candidate in candidates:
        key = candidate["product_key"]
        current_best = best_by_key.get(key)

        if current_best is None:
            best_by_key[key] = candidate
            continue

        if float(candidate.get(score_name) or 0.0) > float(current_best.get(score_name) or 0.0):
            best_by_key[key] = candidate

    return list(best_by_key.values())


def analyze_user_intent_node(state: FashionState) -> FashionState:
    user_query = (state.get("user_query") or "").strip()
    user_image = state.get("user_image")

    has_text = bool(user_query)
    has_image = user_image is not None

    if has_text and has_image:
        input_type = "text_and_image"
    elif has_image:
        input_type = "image_only"
    elif has_text:
        input_type = "text_only"
    else:
        input_type = "empty"

    lower_query = user_query.lower()

    refinement_keywords = [
        "another",
        "another one",
        "show another",
        "show me another",
        "other option",
        "other options",
        "different one",
        "different option",
        "more options",
        "next",
        "next one",
    ]

    if input_type == "empty":
        user_intent = "missing_input"
    elif any(keyword in lower_query for keyword in refinement_keywords):
        user_intent = "refine_previous_search"
    elif has_image and _is_vague_text(user_query):
        user_intent = "find_similar"
    else:
        user_intent = "product_search"

    return {
        **state,
        "has_text": has_text,
        "has_image": has_image,
        "input_type": input_type,
        "user_intent": user_intent,
        "debug_trace": _add_trace(
            state,
            "analyze_user_intent",
            {
                "input_type": input_type,
                "user_intent": user_intent,
                "has_text": has_text,
                "has_image": has_image,
            },
        ),
    }


def _select_dynamic_alpha(retrieval_strategy: str) -> float:
    """
    Alpha controls the final merge balance.

    In this project:
    - alpha close to 1.0 means stronger image score weight.
    - alpha close to 0.0 means stronger text score weight.
    """
    strategy_to_alpha = {
        "none": 0.0,
        "text_only": 0.0,
        "image_only": 1.0,
        "image_priority_multimodal": 0.9,
        "multimodal": 0.6,
    }

    return strategy_to_alpha.get(retrieval_strategy, 0.6)


def decide_retrieval_strategy_node(state: FashionState) -> FashionState:
    input_type = state.get("input_type", "empty")
    user_intent = state.get("user_intent", "product_search")

    if input_type == "empty":
        retrieval_strategy = "none"
    elif input_type == "image_only":
        retrieval_strategy = "image_only"
    elif input_type == "text_only":
        retrieval_strategy = "text_only"
    elif input_type == "text_and_image" and user_intent == "find_similar":
        retrieval_strategy = "image_priority_multimodal"
    else:
        retrieval_strategy = "multimodal"

    dynamic_alpha = _select_dynamic_alpha(retrieval_strategy)

    return {
        **state,
        "retrieval_strategy": retrieval_strategy,
        "alpha": dynamic_alpha,
        "debug_trace": _add_trace(
            state,
            "decide_retrieval_strategy",
            {
                "retrieval_strategy": retrieval_strategy,
                "dynamic_alpha": dynamic_alpha,
            },
        ),
    }


def generate_search_description_node(state: FashionState) -> FashionState:
    chatbot = get_chatbot()

    user_query = state.get("user_query")
    user_image = state.get("user_image")

    if state.get("user_intent") == "missing_input":
        return {
            **state,
            "search_description": "",
            "error": "No user query or image was provided.",
        }

    try:
        if state.get("user_intent") == "refine_previous_search":
            conversation_history = chatbot.conversation
        else:
            conversation_history = []

        search_description = chatbot.reasoning_service.generate_search_description(
            conversation_history=conversation_history,
            user_query=user_query,
            user_image=user_image,
        )

        return {
            **state,
            "search_description": search_description,
            "debug_trace": _add_trace(
                state,
                "generate_search_description",
                {
                    "search_description": search_description,
                    "used_conversation_history": len(conversation_history) > 0,
                },
            ),
        }

    except Exception as exc:
        fallback_description = user_query or ""

        return {
            **state,
            "search_description": fallback_description,
            "error": f"Search description generation failed: {str(exc)}",
            "debug_trace": _add_trace(
                state,
                "generate_search_description",
                {"fallback_description": fallback_description},
            ),
        }


def extract_structured_query_node(state: FashionState) -> FashionState:
    """
    Step 3 node: extract product attributes and store them in graph state.

    We use search_description first because it may include information generated
    from the image by the reasoning model. If it is empty, we fall back to the
    original user_query.
    """
    source_text = state.get("search_description") or state.get("user_query") or ""
    structured_query = _extract_structured_query(source_text)

    return {
        **state,
        "structured_query": structured_query,
        "debug_trace": _add_trace(
            state,
            "extract_structured_query",
            {
                "structured_query": structured_query,
            },
        ),
    }


def retrieve_image_candidates_node(state: FashionState) -> FashionState:
    """Step 4: run image retrieval as its own path."""
    chatbot = get_chatbot()

    retrieval_strategy = state.get("retrieval_strategy", "multimodal")
    user_image = state.get("user_image")
    retrieval_top_k = state.get("retrieval_top_k", 30)

    image_enabled = retrieval_strategy in {
        "image_only",
        "image_priority_multimodal",
        "multimodal",
    }

    if not image_enabled or user_image is None:
        image_candidates = []
    else:
        image_scores, image_indices = chatbot.retrieval_service.search_by_image(
            user_image,
            top_k=retrieval_top_k,
        )

        image_candidates = [
            _candidate_from_index(
                idx=idx,
                score=score,
                score_name="image_score",
                master_df=chatbot.master_df,
                retrieval_service=chatbot.retrieval_service,
            )
            for score, idx in zip(image_scores, image_indices)
            if int(idx) != -1
        ]

        image_candidates = _deduplicate_candidates(image_candidates, "image_score")

    return {
        **state,
        "image_candidates": image_candidates,
        "debug_trace": _add_trace(
            state,
            "retrieve_image_candidates",
            {
                "image_path_enabled": image_enabled,
                "image_candidate_count": len(image_candidates),
            },
        ),
    }


def retrieve_text_candidates_node(state: FashionState) -> FashionState:
    """Step 4: run text retrieval as its own path."""
    chatbot = get_chatbot()

    retrieval_strategy = state.get("retrieval_strategy", "multimodal")
    search_description = state.get("search_description") or ""
    retrieval_top_k = state.get("retrieval_top_k", 30)

    text_enabled = retrieval_strategy in {
        "text_only",
        "image_priority_multimodal",
        "multimodal",
    }

    if not text_enabled or not search_description:
        text_candidates = []
    else:
        text_scores, text_indices = chatbot.retrieval_service.search_by_text(
            search_description,
            top_k=retrieval_top_k,
        )

        text_candidates = [
            _candidate_from_index(
                idx=idx,
                score=score,
                score_name="text_score",
                master_df=chatbot.master_df,
                retrieval_service=chatbot.retrieval_service,
            )
            for score, idx in zip(text_scores, text_indices)
            if int(idx) != -1
        ]

        text_candidates = _deduplicate_candidates(text_candidates, "text_score")

    return {
        **state,
        "text_candidates": text_candidates,
        "debug_trace": _add_trace(
            state,
            "retrieve_text_candidates",
            {
                "text_path_enabled": text_enabled,
                "text_candidate_count": len(text_candidates),
            },
        ),
    }


def merge_candidates_node(state: FashionState) -> FashionState:
    """
    Step 4: merge image and text candidates by product identity.

    We do NOT merge by raw FAISS index here. The product key is image_id + item_id.
    This makes the merge safer if image/text indexes ever differ in row ordering.
    """
    chatbot = get_chatbot()

    image_candidates = state.get("image_candidates", [])
    text_candidates = state.get("text_candidates", [])
    display_top_k = state.get("display_top_k", 6)
    retrieval_strategy = state.get("retrieval_strategy", "multimodal")
    alpha = state.get("alpha", _select_dynamic_alpha(retrieval_strategy))

    image_by_key = {candidate["product_key"]: candidate for candidate in image_candidates}
    text_by_key = {candidate["product_key"]: candidate for candidate in text_candidates}
    all_keys = set(image_by_key.keys()).union(text_by_key.keys())

    both_paths_active = bool(image_candidates) and bool(text_candidates)
    merged_candidates = []

    for key in all_keys:
        image_candidate = image_by_key.get(key)
        text_candidate = text_by_key.get(key)

        base_candidate = dict(image_candidate or text_candidate)

        image_score = None
        text_score = None

        if image_candidate is not None:
            image_score = image_candidate.get("image_score")

        if text_candidate is not None:
            text_score = text_candidate.get("text_score")

        if both_paths_active:
            combined_score = alpha * float(image_score or 0.0) + (1.0 - alpha) * float(text_score or 0.0)
        elif image_score is not None:
            combined_score = float(image_score)
        elif text_score is not None:
            combined_score = float(text_score)
        else:
            combined_score = 0.0

        base_candidate["image_score"] = image_score
        base_candidate["text_score"] = text_score
        base_candidate["combined_score"] = combined_score
        base_candidate["score"] = combined_score
        base_candidate["matched_image_path"] = image_candidate is not None
        base_candidate["matched_text_path"] = text_candidate is not None

        merged_candidates.append(base_candidate)

    merged_candidates = sorted(
        merged_candidates,
        key=lambda candidate: candidate["combined_score"],
        reverse=True,
    )

    products = merged_candidates[:display_top_k]
    product_images = chatbot.retrieval_service.load_product_images(products)

    return {
        **state,
        "merged_candidates": merged_candidates,
        "products": products,
        "product_images": product_images,
        "mode": "separate_retrieval_paths",
        "debug_trace": _add_trace(
            state,
            "merge_candidates",
            {
                "retrieval_strategy": retrieval_strategy,
                "alpha": alpha,
                "image_candidate_count": len(image_candidates),
                "text_candidate_count": len(text_candidates),
                "merged_candidate_count": len(merged_candidates),
                "returned_product_count": len(products),
            },
        ),
    }


def update_conversation_node(state: FashionState) -> FashionState:
    chatbot = get_chatbot()

    user_query = state.get("user_query")
    search_description = state.get("search_description")
    structured_query = state.get("structured_query", {})

    chatbot.conversation.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    chatbot.conversation.append(
        {
            "role": "assistant",
            "content": search_description,
            "structured_query": structured_query,
        }
    )

    return {
        **state,
        "conversation": chatbot.conversation,
        "debug_trace": _add_trace(
            state,
            "update_conversation",
            {"conversation_length": len(chatbot.conversation)},
        ),
    }


def build_fashion_graph():
    graph = StateGraph(FashionState)

    graph.add_node("analyze_user_intent", analyze_user_intent_node)
    graph.add_node("decide_retrieval_strategy", decide_retrieval_strategy_node)
    graph.add_node("generate_search_description", generate_search_description_node)
    graph.add_node("extract_structured_query", extract_structured_query_node)
    graph.add_node("retrieve_image_candidates", retrieve_image_candidates_node)
    graph.add_node("retrieve_text_candidates", retrieve_text_candidates_node)
    graph.add_node("merge_candidates", merge_candidates_node)
    graph.add_node("update_conversation", update_conversation_node)

    graph.add_edge(START, "analyze_user_intent")
    graph.add_edge("analyze_user_intent", "decide_retrieval_strategy")
    graph.add_edge("decide_retrieval_strategy", "generate_search_description")
    graph.add_edge("generate_search_description", "extract_structured_query")
    graph.add_edge("extract_structured_query", "retrieve_image_candidates")
    graph.add_edge("retrieve_image_candidates", "retrieve_text_candidates")
    graph.add_edge("retrieve_text_candidates", "merge_candidates")
    graph.add_edge("merge_candidates", "update_conversation")
    graph.add_edge("update_conversation", END)

    return graph.compile()


fashion_graph = build_fashion_graph()
