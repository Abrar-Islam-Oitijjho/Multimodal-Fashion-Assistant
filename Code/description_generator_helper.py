import torch
import textwrap


class custom_prompt:

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def build_conv(self, user_text, user_image=True):

        conversation = [
            {
                "role": "user",
                "content": []
            }
        ]
    
        if user_image:
            conversation[0]["content"].append({"type": "image"})
    
        conversation[0]["content"].append({"type": "text", "text": user_text})
    
        return conversation

    def description_of_image(self, image):
        
        user_text = (
            "Describe ONLY the main clothing item shown in the image. "
            "Ignore the person and ignore the background. "
            "Do NOT mention other objects. "
            "Describe only the clothing item's color, print style, material, stripes, and notable features."
        )
        
        conversation = self.build_conv(user_text)
        text_prompt = self.processor.apply_chat_template(conversation,add_generation_prompt=True)
        
        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=128)
        
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        del inputs
        del output_ids
        torch.cuda.empty_cache()
        
        return output_text

    def attributes_of_image(self, description_text):

        prompt = textwrap.dedent(f"""
        Extract product attributes from this description in JSON format:
        Description: "{description_text}"
        Return: color, material, fit, fabric, logo, stripes, neckline, print style, any other relevant features.
        """)

        inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=50)
        
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        attributes = {}
        for line in output_text[0].split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)  # split at first colon
                attributes[key.strip()] = value.strip()

        del inputs
        del output_ids
        torch.cuda.empty_cache()
        
        return attributes