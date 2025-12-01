import os
import json
from PIL import Image


class data_preprocessing():

    def load_annotations(self, file_path):
    
        rows = []
    
        with open(file_path, "r") as f:
            data = json.load(f)
            
            file = os.path.basename(file_path)
            image_id = file[:6]
            pair_id = data.get("pair_id")
            source = data.get("source")
    
            for key, item in data.items():
                if not key.startswith("item"):
                    continue
                    
                category_id = item.get("category_id")
                category_name = item.get("category_name")
                style = item.get("style")
                bbox = item.get("bounding_box")
                landmarks = item.get("landmarks")
                segmentation = item.get("segmentation")
    
                scale = item.get("scale")
                occlusion = item.get("occlusion")
                zoom_in = item.get("zoom_in")
                viewpoint = item.get("viewpoint")
    
                row = {
                    "image_id": image_id,
                    "pair_id": pair_id,
                    "source": source,
                    "item_id": key,                      # item1, item2, ...
                    "category_id": category_id,
                    "category_name": category_name,
                    "style": style,
                    "bbox": bbox,
                    "landmarks": landmarks,
                    "segmentation": segmentation,
                    "scale": scale,
                    "occlusion": occlusion,
                    "zoom_in": zoom_in,
                    "viewpoint": viewpoint
                    }
    
                rows.append(row)
    
        return rows
    
    def crop_image(self, directory, df):
    
        parent = os.path.dirname(directory)
        save_folder = os.path.join(parent, "cropped_image")
        os.makedirs(save_folder, exist_ok=True)
        
        for idx, row in df.iterrows():
        
            image_id = row['image_id']
            bbox = row['bbox'] 
            item_id = row['item_id']
        
            if bbox is None:
                continue
        
            x1, y1, x2, y2 = bbox
            
            image_path = os.path.join(directory, image_id + ".jpg")
            if not os.path.exists(image_path):
                continue
        
            img = Image.open(image_path)
            cropped_img = img.crop((x1, y1, x2, y2))
        
            # Saving cropped image
            cropped_name = f"{image_id}_{item_id}.jpg"
            cropped_img.save(os.path.join(save_folder, cropped_name))   
            
    
    def get_all_images_and_mapping(self, image_directory, df):

        images = []

        for idx, row in df.iterrows():

            image_id = row.image_id
            item_id = row.item_id
            image_path = os.path.join(image_directory, f"{image_id}_{item_id}.jpg")
            
            if not os.path.exists(image_path):
                image_path = os.path.join(image_directory, f"{image_id}_{item_id}.png")  # trying png otherwise skipping
                if not os.path.exists(image_path):
                    continue
            
            pil = Image.open(image_path).convert("RGB")
            images.append(pil)

        return images