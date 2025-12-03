import torch
from PIL import Image
import os
import numpy as np
import faiss
import math
import matplotlib.pyplot as plt

from embeddings_generator_helper import embeddings_generator_and_retreival


class chatbot():

    def __init__(self, reasoning_model, reasoning_processor, retreiving_model, retreiving_processor):
            
        self.reasoning_model = reasoning_model
        self.reasoning_processor = reasoning_processor
        
        self.retreiving_model = retreiving_model
        self.retreiving_processor = retreiving_processor
        
        
    def classifier(self, user_image):
    
        content = [
            {"type": "text", "text":
             "Identify the type of content in this image."
             "If it contains a piece of clothing or fashion item, answer 'clothing'."
             "If it contains only text, handwriting, or a printed list, answer 'text'."
             "Answer with exactly one word, no explanation."},
            {"type": "image"}
        ]    
    
        conversation_followup = [{"role": "user", "content": content}]
    
        text_prompt = self.reasoning_processor.apply_chat_template(conversation_followup, add_generation_prompt=True)
    
        inputs_kwargs = {"text": text_prompt, "return_tensors": "pt"}
    
        if user_image:
            inputs_kwargs["images"] = user_image
    
        inputs = self.reasoning_processor(**inputs_kwargs).to(self.reasoning_model.device)
    
        with torch.no_grad():
            output_ids = self.reasoning_model.generate(**inputs, max_new_tokens=80)
            
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        output_text = self.reasoning_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        del inputs
        del output_ids
        torch.cuda.empty_cache()
        
        return output_text


    def process_shopping_list(self, user_image):

        content = [
            {"type": "text", "text":
             "Read the text. Extract only shopping list items. Return one item per line."},
            {"type": "image"}
        ]    
    
        conversation_followup = [{"role": "user", "content": content}]
    
        text_prompt = self.reasoning_processor.apply_chat_template(conversation_followup, add_generation_prompt=True)
    
        inputs_kwargs = {"text": text_prompt, "return_tensors": "pt"}
    
        if user_image:
            inputs_kwargs["images"] = user_image
    
        inputs = self.reasoning_processor(**inputs_kwargs).to(self.reasoning_model.device)
    
        with torch.no_grad():
            output_ids = self.reasoning_model.generate(**inputs, max_new_tokens=80)
            
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        output_text = self.reasoning_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        del inputs
        del output_ids
        torch.cuda.empty_cache()
    
        items = [
            line.strip().lstrip("-• ").strip()
            for line in output_text.split("\n")
            if line.strip()
        ]
        
        return items
    
    def generate_description(self, conversation, user_query, user_image):
    
        previous_conversation = conversation.copy()
        
        content_list = [
        {"type": "text", "text": (
            "Describe exactly which product the user wants now. "
            "Describe the product's color, print style, material, stripes, size, and notable features."
            )}
            ]

        # Checking if it's a followup conversation or not
        if len(conversation) > 0:
            content_list.insert(0, {"type": "text", "text": f"Previous conversation:\n{previous_conversation}"})
    
        if user_image:
            content_list.insert(1, {"type": "image"})
            
        if user_query:
            content_list.insert(1, {"type": "text", "text": f"Follow-up query: {user_query}"})
    
        conversation_followup = [{"role": "user", "content": content_list}]
        
        text_prompt = self.reasoning_processor.apply_chat_template(conversation_followup, add_generation_prompt=True)
    
        inputs_kwargs = {"text": text_prompt, "return_tensors": "pt"}
    
        if user_image:
            inputs_kwargs["images"] = user_image
    
        inputs = self.reasoning_model(**inputs_kwargs).to(self.reasoning_model.device)
    
        with torch.no_grad():
            output_ids = self.reasoning_model.generate(**inputs, max_new_tokens=80)
        
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        output_text = self.reasoning_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        del inputs
        del output_ids
        torch.cuda.empty_cache()
        
        return output_text
    
    def retreive_index(self, faiss_index_directory, description, user_query, user_image, top_k):

        emb_ = embeddings_generator_and_retreival(self.retreiving_model, self.retreiving_processor)        
        dist_img = idx_img = dist_text = idx_text = np.array([])
    
        img_faiss_index =  faiss.read_index(os.path.join(faiss_index_directory, "faiss_image_siglip2_base.index"))
        text_faiss_index =  faiss.read_index(os.path.join(faiss_index_directory, "faiss_text_siglip2_base.index"))
        top_k = 10

        if user_image:
            dist_img, idx_img = emb_.retrieval_image(user_image, img_faiss_index, top_k)
            dist_img, idx_img = dist_img[0], idx_img[0]
            
        if user_query:
            dist_text, idx_text = emb_.retrieval_text(description[0], text_faiss_index, top_k)
            dist_text, idx_text = dist_text[0], idx_text[0]

        return dist_img, idx_img, dist_text, idx_text
    
    def calculate_combined_score(self, idx_img, dist_img, idx_text, dist_text, alpha):

        img_dict = text_dict = None

        if idx_img.any():
            img_dict = {idx: score for idx, score in zip(idx_img, dist_img)}

        if idx_text.any():
            text_dict = {idx: score for idx, score in zip(idx_text, dist_text)}
        
        all_candidates = set(idx_img).union(set(idx_text))
        
        # Computing combined score for each candidate
        combined_scores = []
        if img_dict and text_dict:
            for idx in all_candidates:
                img_score = img_dict.get(idx, 0)    # if candidate not in top-k, assuming 0
                text_score = text_dict.get(idx, 0) 
                final_score = alpha * img_score + (1 - alpha) * text_score
                combined_scores.append((idx, final_score))
            # Sort by final_score descending
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            combined_scores = dict(combined_scores)
            return combined_scores

        elif img_dict:
            return img_dict

        else:
            return text_dict
        
    
    def retreive_top_products(self, combined_scores, master_df, top_k):
        
        top_k_keys = [k for k, v in sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]
        suggested_images = []
        
        rows = math.ceil(top_k / 3)  # 3 is the max number of images per row
        plt.figure(figsize=(4 * 3, 4 * rows))  # 4 is the width/height of one image in inches for the figure size
    
        for i, idx in enumerate(top_k_keys):
            image_id = master_df.iloc[idx]['image_id']
            item_id = master_df.iloc[idx]['item_id']
            
            image_directory = r"D:\LLM_Project\Multimodel Chatbot\Data\train\cropped_image_unique"
            retreival_path = os.path.join(image_directory, f"{image_id}_{item_id}.jpg")
            image = Image.open(retreival_path).convert("RGB")
            suggested_images.append(image)
            plt.subplot(rows, 3, i + 1)
            plt.imshow(image)
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()
    
        return top_k_keys, suggested_images
    
    
    def comapct_chatbot(self, conversation, faiss_index_directory, master_df, user_query, user_image, top_k, alpha):

        output_text = self.generate_description(conversation=conversation, user_query=user_query, user_image=user_image)
        print(output_text[0])
        
        dist_img, idx_img, dist_text, idx_text = self.retreive_index(faiss_index_directory, output_text, user_query=user_query, 
                                                                       user_image=user_image, top_k=top_k[0])
        
        combined_scores = self.calculate_combined_score(idx_img, dist_img, idx_text, dist_text, alpha)
        #print(combined_scores)
        
        top_k_keys, suggested_images = self.retreive_top_products(combined_scores, master_df, top_k=top_k[1])
        
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": user_query}]
        })
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": output_text[0]}]
        })
    
        result = {}
        result['conversation'] = conversation
        result['output_text'] = output_text
        result['top_k_keys'] = top_k_keys
        result['suggested_images'] = suggested_images
        
        return result