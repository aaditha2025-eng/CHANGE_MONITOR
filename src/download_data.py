import os
from datasets import load_dataset
import numpy as np
from PIL import Image

def download_sample_dataset():
    print("Connecting to HuggingFace Datasets...")
    try:
        # Load a popular Change Detection dataset (OSCD - Onera Satellite Change Detection)
        # We only download the first 50 pairs to save time for the hackathon
        print("Downloading OSCD Dataset (this may take a minute)...")
        ds = load_dataset("blanchon/OSCD_RGB", split="train[:50]")
        
        os.makedirs("dataset/before", exist_ok=True)
        os.makedirs("dataset/after", exist_ok=True)
        os.makedirs("dataset/masks", exist_ok=True)
        
        print(f"Successfully downloaded {len(ds)} image pairs. Formatting folders...")
        
        for i, example in enumerate(ds):
            # Extract images from the HuggingFace dictionary
            # Fallback to standard keys if OSCD uses different names
            b_key = 'image_1' if 'image_1' in example else 'image1' if 'image1' in example else 'before' if 'before' in example else list(example.keys())[0]
            a_key = 'image_2' if 'image_2' in example else 'image2' if 'image2' in example else 'after' if 'after' in example else list(example.keys())[1]
            m_key = 'mask' if 'mask' in example else 'label' if 'label' in example else list(example.keys())[2]

            before_img = example[b_key]
            after_img = example[a_key]
            mask_img = example[m_key]
            
            # Save images
            before_img.save(f"dataset/before/img_{i:03d}.jpg")
            after_img.save(f"dataset/after/img_{i:03d}.jpg")
            
            # Masks must be PNG to preserve pure black/white pixels
            if not isinstance(mask_img, Image.Image):
                mask_img = Image.fromarray(np.array(mask_img))
            mask_img.save(f"dataset/masks/img_{i:03d}.png")
            
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/50 images...")
                
        print("\n✅ Dataset successfully downloaded and organized!")
        print("You can now run: python src/train.py --data_dir ./dataset --epochs 10")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("If HuggingFace times out, you may need to use Kaggle instead.")

if __name__ == "__main__":
    download_sample_dataset()
