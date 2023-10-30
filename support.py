from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
from models.tokenization_bert import BertTokenizer

import torch
from torch import nn
from torchvision import transforms

import json

class VL_Transformer_ITM(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 config_bert = '',
                 add_size = False
                 ):
        super().__init__()
    
        bert_config = BertConfig.from_json_file(config_bert)

        self.visual_encoder = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)   
        
        self.itm_head = nn.Linear(768, 2) 

        self.add_size = add_size
        if add_size:
            
            self.size_pred_head = nn.Linear(768, 1)
        
    def forward(self, image, text):
        image_embeds = self.visual_encoder(image) 

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        output = self.text_encoder(text.input_ids, 
                                attention_mask = text.attention_mask,
                                encoder_hidden_states = image_embeds,
                                encoder_attention_mask = image_atts,      
                                return_dict = True,
                               )     
           
        vl_embeddings = output.last_hidden_state[:,0,:]
        vl_output = self.itm_head(vl_embeddings) 
        
        if self.add_size:
            size_output = self.size_pred_head(vl_embeddings)
            return vl_output, size_output
        return vl_output
    
import re

def pre_caption(caption,max_words=30):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])            
    return caption


from PIL import Image

import cv2
import numpy as np

from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt

def getAttMap(img, attMap, blur = True, overlap = True, threshold = 0.0):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
    
    

    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
        
    mask_threshold = attMap >= threshold
    attMap = attMap*mask_threshold
    
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap


normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transform = transforms.Compose([
    transforms.Resize((384,384),interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
]) 


import torch
import torch.nn.functional as F
from torch import nn

def get_gcam(model_new, image_new, text_input):
    block_num = 8
    bs = 1
    vl_output = model_new(image_new, text_input)
    fmaps=model_new.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map()
    output = vl_output[:bs]

    one_hot = torch.zeros_like(output)
    one_hot[:,1]=1
    #             print('one_hot shape',one_hot.shape)
    grad_wrt_act = torch.autograd.grad(outputs=output, inputs=fmaps, grad_outputs=one_hot, \
                                                   create_graph=True)[0]
    mask = text_input.attention_mask.view(text_input.attention_mask.size(0),1,-1,1,1)

    fmaps = fmaps[:, :, :, 1:].reshape(bs, 12, -1, 24, 24) * mask
    #             print('grad_wrt_act shape',grad_wrt_act.shape)
    grad_wrt_act = grad_wrt_act[:, :, :, 1:].clamp(0).reshape(bs, 12, -1, 24, 24) * mask

    gradcam = fmaps * grad_wrt_act
    gradcam = gradcam.mean(1).mean(1)
    B, H, W = gradcam.shape
    gcam_raw = gradcam
    gcam_raw = gcam_raw.view(B, -1)
    gcam_raw -= gcam_raw.min(dim=1, keepdim=True)[0]
    gcam_raw /= (gcam_raw.max(dim=1, keepdim=True)[0]+0.0000001)
    gcam_raw = gcam_raw.view(B, H, W)
    # print(gcam)
    gradcam = F.relu(gradcam)


    gradcam = gradcam.view(B, -1)
    gradcam -= gradcam.min(dim=1, keepdim=True)[0]
    gradcam /= (gradcam.max(dim=1, keepdim=True)[0]+0.0000001)
    gradcam_for_loss = gradcam.view(B, H, W)
    return gradcam_for_loss
    # loss_gcam_masked = 1-F.cosine_similarity(gradcam.view(1,-1), mask_query_interp.view(1,-1))
    
def get_gcam_train(model_new, image_new, text_input):
    block_num = 8
    bs = 1
    vl_output = model_new(image_new, text_input)
    fmaps=model_new.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map()
    output = vl_output[:bs]

    one_hot = torch.zeros_like(output)
    one_hot[:,1]=1
    #             print('one_hot shape',one_hot.shape)
    grad_wrt_act = torch.autograd.grad(outputs=output, inputs=fmaps, grad_outputs=one_hot, \
                                                   create_graph=True)[0]
    mask = text_input.attention_mask.view(text_input.attention_mask.size(0),1,-1,1,1)

    fmaps = fmaps[:, :, :, 1:].reshape(bs, 12, -1, 24, 24) * mask
    #             print('grad_wrt_act shape',grad_wrt_act.shape)
    grad_wrt_act = grad_wrt_act[:, :, :, 1:].clamp(0).reshape(bs, 12, -1, 24, 24) * mask

    gradcam = fmaps * grad_wrt_act
    gradcam = gradcam.mean(1).mean(1)
    B, H, W = gradcam.shape
    gcam_raw = gradcam
    gcam_raw = gcam_raw.view(B, -1)
    gcam_raw -= gcam_raw.min(dim=1, keepdim=True)[0]
    gcam_raw /= (gcam_raw.max(dim=1, keepdim=True)[0]+0.0000001)
    gcam_raw = gcam_raw.view(B, H, W)
    # print(gcam)
    gradcam = F.relu(gradcam)


    gradcam = gradcam.view(B, -1)
    gradcam -= gradcam.min(dim=1, keepdim=True)[0]
    gradcam /= (gradcam.max(dim=1, keepdim=True)[0]+0.0000001)
    gradcam_for_loss = gradcam.view(B, H, W)
    
    gradcam_for_loss = gradcam_for_loss.view(B,1,H,W)
    gradcam_for_loss = F.interpolate(
        gradcam_for_loss, (256,256), mode="bilinear", align_corners=False
    ).squeeze()
    return gradcam_for_loss
    # loss_gcam_masked = 1-F.cosine_similarity(gradcam.view(1,-1), mask_query_interp.view(1,-1))
    
