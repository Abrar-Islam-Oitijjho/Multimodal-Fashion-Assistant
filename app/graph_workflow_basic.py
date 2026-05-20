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

    search_description: str
    products: list[dict]
    product_images: list[Any]
    conversation: list[dict]
    mode: str
    error: Optional[str]


def generate_search_description_node(state: FashionState) -> FashionState:
    chatbot = get_chatbot()

    user_query = state.get("user_query")
    user_image = state.get("user_image")

    try:
        search_description = chatbot.reasoning_service.generate_search_description(
            conversation_history=chatbot.conversation,
            user_query=user_query,
            user_image=user_image,
        )

        return {
            **state,
            "search_description": search_description,
        }

    except Exception as exc:
        return {
            **state,
            "search_description": user_query or "",
            "error": f"Search description generation failed: {str(exc)}",
        }


def retrieve_products_node(state: FashionState) -> FashionState:
    chatbot = get_chatbot()

    search_description = state.get("search_description")
    user_image = state.get("user_image")

    retrieval_top_k = state.get("retrieval_top_k", 10)
    display_top_k = state.get("display_top_k", 6)
    alpha = state.get("alpha", 0.4)

    combined_scores = chatbot.retrieval_service.retrieve(
        query_text=search_description,
        query_image=user_image,
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
    }


def build_fashion_graph():
    graph = StateGraph(FashionState)

    graph.add_node("generate_search_description", generate_search_description_node)
    graph.add_node("retrieve_products", retrieve_products_node)
    graph.add_node("update_conversation", update_conversation_node)

    graph.add_edge(START, "generate_search_description")
    graph.add_edge("generate_search_description", "retrieve_products")
    graph.add_edge("retrieve_products", "update_conversation")
    graph.add_edge("update_conversation", END)

    return graph.compile()


fashion_graph = build_fashion_graph()