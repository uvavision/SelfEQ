import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
import copy

import numpy as np
import torch
     

class pretrain_dataset_coco(Dataset):
    def __init__(self, ann_file, transform, image_root = '', max_words=50, phrase_input=False, add_gcam = False, mask_all = False, mask_size = 16):       
        self.image_root = image_root
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform #[0]
        self.max_words = max_words
        self.phrase_input = phrase_input

        self.sigma = -1
        self.add_gcam = add_gcam
        self.mask_all = mask_all
        
        self.mask_size = mask_size
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    

        ann = self.ann[index]
        if self.phrase_input:
            if type(ann['phrase_chunks']) == list:
                caption = pre_caption(random.choice(ann['phrase_chunks']), self.max_words)
            else:
                caption = pre_caption(ann['phrase_chunks'], self.max_words)
        else:
            if type(ann['phrase_chunks']) == list:
                caption = pre_caption(random.choice(ann['phrase_chunks']), self.max_words)
            else:
                caption = pre_caption(ann['phrase_chunks'], self.max_words)
      
    
        image = Image.open(os.path.join(self.image_root,ann['image'])).convert('RGB') 
        w,h = image.size
        #image = self.transform(image)
        image = self.transform(image = np.array(image))['image']

        gt_mask_indicator = 1.0
        
        return image, caption, torch.LongTensor([gt_mask_indicator])
            
        # else:
        #     # mask_query_interp = None
        #     # gt_mask_indicator = None
        #     image = self.transform(image)
        #     return image, caption
                
        
        

            
# class pretrain_dataset(Dataset):
#     def __init__(self, ann_file, transform, image_root = '', max_words=50, phrase_input=False, add_gcam = False, mask_all = False, mask_size = 16):       
#         self.image_root = image_root
#         self.ann = []
#         for f in ann_file:
#             self.ann += json.load(open(f,'r'))
#         self.transform = transform
#         self.max_words = max_words
#         self.phrase_input = phrase_input
        
        
#     def __len__(self):
#         return len(self.ann)
    

#     def __getitem__(self, index):    

#         ann = self.ann[index]

#         if type(ann['caption']) == list:
#             caption = pre_caption(random.choice(ann['caption']), self.max_words)
#         else:
#             caption = pre_caption(ann['caption'], self.max_words)
      
    
#         image = Image.open(os.path.join(self.image_root,ann['image'])).convert('RGB') 
#         w,h = image.size
#         image = self.transform(image)
            
#         return image, caption
            
#         # else:
#         #     # mask_query_interp = None
#         #     # gt_mask_indicator = None
#         #     image = self.transform(image)
#         #     return image, caption
                
        
        

            
    
