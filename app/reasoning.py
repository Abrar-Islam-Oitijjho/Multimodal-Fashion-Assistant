import torch


class ReasoningService:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def _generate(self, content: list[dict], max_new_tokens: int = 100) -> str:
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        inputs_kwargs = {
            "text": text_prompt,
            "return_tensors": "pt",
        }

        has_image = any(item.get("type") == "image" for item in content)

        image_items = [
            item.get("image")
            for item in content
            if item.get("type") == "image" and item.get("image") is not None
        ]

        if has_image and image_items:
            inputs_kwargs["images"] = image_items

        inputs = self.processor(**inputs_kwargs).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        del inputs
        del output_ids

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_text.strip()

    def classify_image(self, image) -> str:
        content = [
            {
                "type": "text",
                "text": (
                    "Identify the type of content in this image. "
                    "If it contains a piece of clothing or fashion item, answer clothing. "
                    "If it contains only text, handwriting, or a printed list, answer text. "
                    "Answer with exactly one word."
                ),
            },
            {
                "type": "image",
                "image": image,
            },
        ]

        result = self._generate(content, max_new_tokens=20)
        result = result.lower().strip()

        if "text" in result:
            return "text"

        if "clothing" in result:
            return "clothing"

        return result

    def process_shopping_list(self, image) -> list[str]:
        content = [
            {
                "type": "text",
                "text": (
                    "Read the text in the image. "
                    "Extract only shopping list items. "
                    "Return one item per line."
                ),
            },
            {
                "type": "image",
                "image": image,
            },
        ]

        output_text = self._generate(content, max_new_tokens=120)

        items = [
            line.strip().lstrip("-•1234567890. ").strip()
            for line in output_text.split("\n")
            if line.strip()
        ]

        return items

    def generate_search_description(
        self,
        conversation_history: list[dict],
        user_query: str | None = None,
        user_image=None,
    ) -> str:
        content = []

        if conversation_history:
            content.append(
                {
                    "type": "text",
                    "text": f"Previous conversation:\n{conversation_history}",
                }
            )

        if user_query:
            content.append(
                {
                    "type": "text",
                    "text": f"Current user query: {user_query}",
                }
            )

        if user_image is not None:
            content.append(
                {
                    "type": "image",
                    "image": user_image,
                }
            )

        content.append(
            {
                "type": "text",
                "text": (
                    "Describe exactly which fashion product the user wants now. "
                    "Include product type, color, print style, material, stripes, size, "
                    "shape, and notable visual features. "
                    "Return only the product search description."
                ),
            }
        )

        return self._generate(content, max_new_tokens=100)