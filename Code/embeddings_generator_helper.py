import os
import numpy as np
from tqdm import tqdm
import faiss
import torch



class embeddings_generator_and_retreival():
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def embed_image(self, images):
        # Processing the inputs
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            image_emb = self.model.get_image_features(**inputs)
            
        image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
        return image_emb.cpu().numpy().astype(np.float32)  # shape (batch_size, dim) # Converting to CPU numpy float32 for FAISS

    def embed_text(self, descriptions):
        # Processing the inputs
        inputs = self.processor(text=descriptions, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)
            
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
        return text_emb.cpu().numpy().astype(np.float32)  # shape (batch_size, dim) # Converting to CPU numpy float32 for FAISS
    
    def generate_image_embeddings_and_faiss_index(self, parent_directory, all_images, batch_size):

        parent = os.path.dirname(parent_directory)
        image_embeddings_directory = os.path.join(parent, "embeddings_image_siglip2_base")
        os.makedirs(image_embeddings_directory, exist_ok=True)
    
        faiss_image_path = os.path.join(parent, "faiss_image_siglip2_base.index")
    
        batch_no = 1
        num_batches = (len(all_images) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(all_images), batch_size), total=num_batches):
         
            batch_images = all_images[i: i+batch_size]             # Taking a batch according to batch size 
            
            image_emb = self.embed_image(batch_images)
            
            emb_image_path = os.path.join(image_embeddings_directory, f"batch_{batch_no}.npy")
            np.save(emb_image_path, image_emb)
    
            if batch_no == 1:
                image_dim = image_emb.shape[1]
                image_index = faiss.IndexFlatIP(image_dim)
                
            image_index.add(image_emb)

            del image_emb
            torch.cuda.empty_cache()
            
            batch_no += 1
    
        faiss.write_index(image_index, faiss_image_path)

    def generate_text_embeddings_and_faiss_index(self, parent_directory, all_descriptions, batch_size):

        parent = os.path.dirname(parent_directory)
        text_embeddings_directory = os.path.join(parent, "embeddings_text_siglip2_base")
        os.makedirs(text_embeddings_directory, exist_ok=True)
    
        faiss_text_path = os.path.join(parent, "faiss_text_siglip2_base.index")
    
        batch_no = 1
        num_batches = (len(all_descriptions) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(all_descriptions), batch_size), total=num_batches):
         
            batch_texts = all_descriptions[i: i+batch_size].tolist()             
    
            text_emb = self.embed_text(batch_texts)
            
            emb_text_path = os.path.join(text_embeddings_directory, f"batch_{batch_no}.npy")
            np.save(emb_text_path, text_emb)
    
            if batch_no == 1:
                text_dim = text_emb.shape[1]
                text_index = faiss.IndexFlatIP(text_dim)
                
            text_index.add(text_emb)

            del text_emb
            torch.cuda.empty_cache()
            
            batch_no += 1
    
        faiss.write_index(text_index, faiss_text_path)

    def retrieval_image(self, image, faiss_index, top_k):
        
        img_emb = self.embed_image(image)
        dist, idx = faiss_index.search(img_emb, top_k)

        return dist, idx

    def retrieval_text(self, text, faiss_index, top_k):
        
        text_emb = self.embed_text(text)
        dist, idx = faiss_index.search(text_emb, top_k)

        return dist, idx