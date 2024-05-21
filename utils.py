from modules.my_clip.clip_model import CLIPTextModel
from transformers import CLIPTokenizer
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import random

@torch.no_grad()
def get_token_embeds(
    tokens : str|list[str],
    tokenizer : CLIPTokenizer,
    text_encoder: CLIPTextModel,
):
    if isinstance(tokens,list):
        tokens=' '.join(tokens)
    
    token_ids=tokenizer(
        tokens,
        padding="do_not_pad",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(text_encoder.device) # (1,k)
    embeds=text_encoder.get_input_embeddings().weight.data[token_ids[0]] # (k+2,1024)
    return embeds[1:-1] #(k,1024)


@torch.no_grad()
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def freeze_model(model:torch.nn.Module,to_freeze:list[str]):
    for name,params in model.named_parameters():
        for freeze_name in to_freeze:
            if freeze_name in name:
                params.requires_grad_(False)

def unfreeze_model(model:torch.nn.Module,to_unfreeze:list[str]):
    for name,params in model.named_parameters():
        for unfreeze_name in to_unfreeze:
            if unfreeze_name in name:
                params.requires_grad_(True)

def get_model_params(model:torch.nn.Module,names:list[str]):
    res=[]
    for model_name,params in model.named_parameters():
        for name in names:
            if name in model_name:
                res.append(params)
    return res