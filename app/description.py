import textwrap

import torch


class DescriptionGenerator:
    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device

    def build_conversation(self, user_text: str, has_image: bool = True) -> list[dict]:
        conversation = [
            {
                "role": "user",
                "content": [],
            }
        ]

        if has_image:
            conversation[0]["content"].append({"type": "image"})

        conversation[0]["content"].append(
            {
                "type": "text",
                "text": user_text,
            }
        )

        return conversation

    def describe_image(self, image) -> str:
        user_text = (
            "Describe ONLY the main clothing item shown in the image. "
            "Ignore the person and ignore the background. "
            "Do NOT mention other objects. "
            "Describe only the clothing item's color, print style, material, "
            "stripes, and notable features."
        )

        conversation = self.build_conversation(user_text, has_image=True)

        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=128)

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

    def extract_attributes(self, description_text: str) -> dict:
        prompt = textwrap.dedent(
            f"""
            Extract product attributes from this description in JSON-style format.

            Description: "{description_text}"

            Return these fields:
            color, material, fit, fabric, logo, stripes, neckline, print style,
            and any other relevant product features.
            """
        )

        inputs = self.processor(
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=80)

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        attributes = {}

        for line in output_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                attributes[key.strip()] = value.strip()

        del inputs
        del output_ids

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return attributes