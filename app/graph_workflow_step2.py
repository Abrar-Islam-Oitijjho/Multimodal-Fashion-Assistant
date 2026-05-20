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


def retrieve_products_node(state: FashionState) -> FashionState:
    chatbot = get_chatbot()

    search_description = state.get("search_description")
    user_image = state.get("user_image")
    retrieval_strategy = state.get("retrieval_strategy", "multimodal")

    retrieval_top_k = state.get("retrieval_top_k", 10)
    display_top_k = state.get("display_top_k", 6)
    alpha = state.get("alpha", 0.4)

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
        # multimodal and image_priority_multimodal both use image + text for now.
        # Step 2 will tune alpha based on the strategy.
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
                "retrieved_count": len(products),
            },
        ),
    }


def update_conversation_node(state: FashionState) -> FashionState:
    chatbot = get_chatbot()

    user_query = state.get("user_query")
    search_description = state.get("search_description")

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
    graph.add_node("retrieve_products", retrieve_products_node)
    graph.add_node("update_conversation", update_conversation_node)

    graph.add_edge(START, "analyze_user_intent")
    graph.add_edge("analyze_user_intent", "decide_retrieval_strategy")
    graph.add_edge("decide_retrieval_strategy", "generate_search_description")
    graph.add_edge("generate_search_description", "retrieve_products")
    graph.add_edge("retrieve_products", "update_conversation")
    graph.add_edge("update_conversation", END)

    return graph.compile()


fashion_graph = build_fashion_graph()
