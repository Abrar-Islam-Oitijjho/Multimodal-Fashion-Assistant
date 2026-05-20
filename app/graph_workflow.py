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
    # direct negative forms
    r"\bwithout\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
    r"\bnot\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
    r"\bno\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",

    # alternative negative words
    r"\bexcluding\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
    r"\bexclude\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
    r"\bavoid\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
    r"\bremove\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
    r"\bexcept\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",
    r"\bother than\s+([a-zA-Z\s-]+?)(?=\s+(?:and|for|with|but|in)\b|,|$)",

    # longer natural forms
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
    
    # another product
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
    "next one"
    
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
    Alpha controls the balance between image and text retrieval.

    In this project:
    - alpha close to 1.0 means stronger image retrieval.
    - alpha close to 0.0 means stronger text retrieval.
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


def retrieve_products_node(state: FashionState) -> FashionState:
    chatbot = get_chatbot()

    search_description = state.get("search_description")
    user_image = state.get("user_image")
    retrieval_strategy = state.get("retrieval_strategy", "multimodal")

    retrieval_top_k = state.get("retrieval_top_k", 10)
    display_top_k = state.get("display_top_k", 6)
    alpha = state["alpha"]

    if retrieval_strategy == "none":
        return {
            **state,
            "products": [],
            "product_images": [],
            "mode": "single_query",
        }

    if retrieval_strategy == "text_only":
        query_text = search_description
        query_image = None
    elif retrieval_strategy == "image_only":
        query_text = None
        query_image = user_image
    else:
        query_text = search_description
        query_image = user_image

    combined_scores = chatbot.retrieval_service.retrieve(
        query_text=query_text,
        query_image=query_image,
        top_k=retrieval_top_k,
        alpha=alpha,
    )

    products = chatbot.retrieval_service.get_top_products(
        combined_scores=combined_scores,
        master_df=chatbot.master_df,
        top_k=display_top_k,
    )

    product_images = chatbot.retrieval_service.load_product_images(products)

    return {
        **state,
        "products": products,
        "product_images": product_images,
        "mode": "single_query",
        "debug_trace": _add_trace(
            state,
            "retrieve_products",
            {
                "retrieval_strategy": retrieval_strategy,
                "alpha": alpha,
                "structured_query": state.get("structured_query", {}),
                "retrieved_count": len(products),
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
    graph.add_node("retrieve_products", retrieve_products_node)
    graph.add_node("update_conversation", update_conversation_node)

    graph.add_edge(START, "analyze_user_intent")
    graph.add_edge("analyze_user_intent", "decide_retrieval_strategy")
    graph.add_edge("decide_retrieval_strategy", "generate_search_description")
    graph.add_edge("generate_search_description", "extract_structured_query")
    graph.add_edge("extract_structured_query", "retrieve_products")
    graph.add_edge("retrieve_products", "update_conversation")
    graph.add_edge("update_conversation", END)

    return graph.compile()


fashion_graph = build_fashion_graph()
