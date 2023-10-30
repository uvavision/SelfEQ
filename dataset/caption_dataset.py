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
     

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root = '', max_words=50, phrase_input=False, add_gcam = False, mask_all = False, mask_size = 16):       
        self.image_root = image_root
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.phrase_input = phrase_input

        if not add_gcam:
            self.transform = transform[0]
            self.boximg_transform = transform[1]
        self.sigma = -1
        self.add_gcam = add_gcam
        self.mask_all = mask_all
        
        self.mask_size = mask_size
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    

        ann = self.ann[index]
        if self.phrase_input:
            box = random.choice(ann['bbox'])
            if type(ann['caption']) == list:
                caption = pre_caption(random.choice(ann['caption']), self.max_words)
            else:
                caption = pre_caption(ann['caption'], self.max_words)
        else:
            if type(ann['caption']) == list:
                caption = pre_caption(random.choice(ann['caption']), self.max_words)
            else:
                caption = pre_caption(ann['caption'], self.max_words)
      
    
        image = Image.open(os.path.join(self.image_root,ann['image'])).convert('RGB') 
        w,h = image.size
            
        if self.add_gcam:
            numpy_img = np.array(image)
            x1,y1,x2,y2 = box
            x1 = min(w-1,max(0,x1))
            y1 = min(h-1,max(0,y1))
            x2 = min(w,max(1,x2))
            y2 = min(h,max(1,y2))
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)

            while len(ann['bbox']) > 1 and (x1 >= x2 or y1 >= y2):
                if type(ann['bbox'][0][0]) == list:
                    # have already sample boxes-phrases pairs before in phrase 5 
                    x1,y1,x2,y2 = random.choice(boxes[0])
                else:
                    x1,y1,x2,y2 = random.choice(ann['bbox'])
                x1 = min(w-1,max(0,x1))
                y1 = min(h-1,max(0,y1))
                x2 = min(w,max(1,x2))
                y2 = min(h,max(1,y2))
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            
            gt_mask = np.zeros([numpy_img.shape[0],numpy_img.shape[1]])
            gt_mask[y1:y2,x1:x2] = 1
            gt_mask_indicator = 1.0
            if self.mask_all:
                if type(ann['bbox'][0][0]) == list:
                    box_list = boxes[0]
                else:
                    box_list = ann['bbox']
                        
                # get max and min for all of them
                all_box_x_min,all_box_x_max,all_box_y_min,all_box_y_max = 10000,-1,10000,-1
                for each_box in box_list:
                    x_min, y_min, x_max, y_max = each_box
                    x_min = min(w-1,max(0,x_min))
                    y_min = min(h-1,max(0,y_min))
                    x_max = min(w,max(1,x_max))
                    y_max = min(h,max(1,y_max))
                    x_min, y_min, x_max, y_max = int(x_min),int(y_min),int(x_max),int(y_max)
                    gt_mask[y_min:y_max,x_min:x_max] = 1
            
            box = [x1,y1,x2,y2]
            box.append(caption)
          
            # box coordinates is useless:
            if box[0] >= box[2]:
                box[2] = box[0] + 1 
            if box[1] >= box[3]:
                box[3] = box[1] + 1
            query_output = self.transform[0](image=numpy_img, mask=gt_mask,bboxes=[box])

            mask_query = query_output['mask']

            mask_query = mask_query.unsqueeze(0).unsqueeze(0)
            mask_query_interp = torch.nn.functional.interpolate(mask_query.type(torch.FloatTensor), self.mask_size, mode='bilinear', align_corners=True).squeeze()
            image = query_output['image']
            return image, caption, mask_query_interp, torch.LongTensor([gt_mask_indicator])
            
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
                
        
        

            
    
