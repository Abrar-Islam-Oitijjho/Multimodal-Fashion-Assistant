import pandas as pd


class FashionChatbot:
    def __init__(
        self,
        reasoning_service,
        retrieval_service,
        master_df: pd.DataFrame,
    ):
        self.reasoning_service = reasoning_service
        self.retrieval_service = retrieval_service
        self.master_df = master_df
        self.conversation = []

    def run_single_query(
        self,
        user_query: str | None = None,
        user_image=None,
        retrieval_top_k: int = 10,
        display_top_k: int = 6,
        alpha: float = 0.4,
    ) -> dict:
        search_description = self.reasoning_service.generate_search_description(
            conversation_history=self.conversation,
            user_query=user_query,
            user_image=user_image,
        )

        combined_scores = self.retrieval_service.retrieve(
            query_text=search_description,
            query_image=user_image,
            top_k=retrieval_top_k,
            alpha=alpha,
        )

        products = self.retrieval_service.get_top_products(
            combined_scores=combined_scores,
            master_df=self.master_df,
            top_k=display_top_k,
        )

        product_images = self.retrieval_service.load_product_images(products)

        self.conversation.append(
            {
                "role": "user",
                "content": user_query,
            }
        )

        self.conversation.append(
            {
                "role": "assistant",
                "content": search_description,
            }
        )

        return {
            "search_description": search_description,
            "products": products,
            "product_images": product_images,
            "conversation": self.conversation,
        }

    def run(
        self,
        user_query: str | None = None,
        user_image=None,
        retrieval_top_k: int = 10,
        display_top_k: int = 6,
        alpha: float = 0.4,
    ) -> dict:
        if user_image is not None:
            image_type = self.reasoning_service.classify_image(user_image)

            if image_type == "text":
                shopping_items = self.reasoning_service.process_shopping_list(
                    user_image,
                )

                results = []

                for item in shopping_items:
                    result = self.run_single_query(
                        user_query=item,
                        user_image=None,
                        retrieval_top_k=retrieval_top_k,
                        display_top_k=3,
                        alpha=alpha,
                    )

                    results.append(result)

                return {
                    "mode": "shopping_list",
                    "shopping_items": shopping_items,
                    "results": results,
                    "conversation": self.conversation,
                }

        result = self.run_single_query(
            user_query=user_query,
            user_image=user_image,
            retrieval_top_k=retrieval_top_k,
            display_top_k=display_top_k,
            alpha=alpha,
        )

        return {
            "mode": "single_query",
            **result,
        }