import faiss
import json
import numpy as np
import os
from PIL import Image
import torch
from tqdm import tqdm

# Monkey-patch torch.load to default map_location='cpu' so MedCLIP's
# hardcoded torch.load calls don't segfault on machines without CUDA.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('map_location', 'cpu')
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Monkey-patch torch.meshgrid to always pass indexing='ij' (the original
# behaviour). Without this, Swin Transformer's relative-position-bias code
# triggers a native crash on macOS with older PyTorch builds.
_orig_meshgrid = torch.meshgrid
def _patched_meshgrid(*tensors, **kwargs):
    kwargs.setdefault('indexing', 'ij')
    return _orig_meshgrid(*tensors, **kwargs)
torch.meshgrid = _patched_meshgrid

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor


def load_clip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] device: {device}")
    encoder_name = MedCLIPVisionModelViT
    print("[DEBUG] creating MedCLIPProcessor")
    feature_extractor = MedCLIPProcessor()
    print("[DEBUG] creating MedCLIPModel")
    clip_model = MedCLIPModel(vision_cls=encoder_name)
    print("[DEBUG] calling from_pretrained")
    clip_model.from_pretrained()
    print("[DEBUG] moving model to device")
    clip_model = clip_model.to(device)
    print("[DEBUG] load_clip_model done")

    return clip_model, feature_extractor, device

def load_datastore(index_path, captions_path):
    """Load the saved FAISS index and corresponding captions."""
    index = faiss.read_index(index_path)
    with open(captions_path, 'r') as f:
        captions = json.load(f)
    return index, captions

def encode_single_image(image_path, model, feature_extractor, device):
    """Encodes a single image into its feature representation."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    image_input = feature_extractor(images=[image], return_tensors='pt').pixel_values.to(device)
    
    with torch.no_grad():
        image_embed = model.encode_image(pixel_values=image_input).cpu().numpy()
    
    return image_embed

def retrieve_info_for_image(index, image_embed, k=7):
    """Retrieve the nearest captions for the given image embedding."""
    xq = image_embed.astype(np.float32)
    faiss.normalize_L2(xq)
    
    # Perform nearest neighbor search
    D, I = index.search(xq, k)
    
    return I[0]  # Return the indices of the nearest neighbors

def get_retrieved_info_for_image(image_path, index_path, captions_path, k=7):
    """Get the retrieved information for a given image."""
    # Load the datastore (index and captions)

    index, captions = load_datastore(index_path, captions_path)

    model, feature_extractor, device = load_clip_model()
    
    # Encode the new image
    image_embed = encode_single_image(image_path, model, feature_extractor, device)
    
    # Retrieve the nearest neighbors
    neighbor_indices = retrieve_info_for_image(index, image_embed, k)
    
    # Extract the retrieved captions
    retrieved_captions = [captions[i] for i in neighbor_indices]
    
    return retrieved_captions


if __name__ == "__main__":
    index_path = "tmp/demo/datastore/kg_nle_index"
    captions_path = "tmp/demo/datastore/kg_nle_index_captions.json"
    image_path = "physionet.org/mimic-cxr-jpg/2.1.0/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
    get_retrieved_info_for_image(image_path, index_path,captions_path, k=7)
