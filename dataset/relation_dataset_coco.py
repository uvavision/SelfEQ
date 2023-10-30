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
     

class relation_dataset_coco(Dataset):
    def __init__(self, ann_file, transform, image_root = '', max_words=50, phrase_input=False):       
        self.image_root = image_root
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.phrase_input = phrase_input
        
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

        template = pre_caption(ann['relation']['group'], self.max_words)
        synonym = ''
        antonym = ''
        hypernym = ''
        meronym = ''
        obj_b=''
        union=''
        isSyn = False
        isAnt = False
        isMer = False
        isHyp = False
        isUni = False

        if len(ann['relation']['synonym'])!=0:
            isSyn = True
            if len(ann['relation']['synonym']) == 1:
                synonym = pre_caption(ann['relation']['synonym'][0], self.max_words)
            else:
                synonym = pre_caption(random.choice(ann['relation']['synonym']), self.max_words)

            synonym = caption.replace(template, synonym)
            

        # exclusion = ann['antonym']
        # print("1 ", exclusion)
        # #print(exclusion, ann['antonym'], ann['exclusion'])
        # exclusion = exclusion.append(ann['exclusion'])
        # print("2 ", exclusion)
        # exclusion = ann['relation']['antonym']+ann['relation']['exclusion']
        # exclusion = ann['relation']['exclusion']
        # if len(exclusion)!=0:
        #     isAnt = True
        #     if len(exclusion) == 1:
        #         antonym = pre_caption(exclusion[0], self.max_words)
        #     else:
        #         antonym = pre_caption(random.choice(exclusion), self.max_words)
            
        # # else:
        # #     antonym = "not " + template
        # if len(ann['relation']['hypernym'])!=0:
        #     isHyp = True
        #     if len(ann['relation']['hypernym']) == 1:
        #         hypernym = pre_caption(ann['relation']['hypernym'][0], self.max_words)
        #     else:
        #         hypernym = pre_caption(random.choice(ann['relation']['hypernym']), self.max_words)
        #     hypernym = caption.replace(template, hypernym)
        # if len(ann['relation']['meronym'])!=0:
        #     isMer = True
        #     if len(ann['relation']['meronym']) == 1:
        #         meronym = pre_caption(ann['relation']['meronym'][0], self.max_words)
        #     else:
        #         meronym = pre_caption(random.choice(ann['relation']['meronym']), self.max_words)
        # if len(ann['relation']['union'])!=0:
        #     isUni = True
        #     if len(ann['relation']['union']) == 1:
        #         union = ann['relation']['union'][0].split(" and ")
        #         #obj_a = pre_caption(union[0], self.max_words)
        #         obj_b = pre_caption(union[1], self.max_words)
        #         union = pre_caption(ann['relation']['union'][0], self.max_words)
        #     else:
        #         union = random.choice(ann['relation']['union'])
        #         union = union.split(" and ")
        #         #obj_a = pre_caption(union[0], self.max_words)
        #         obj_b = pre_caption(union[1], self.max_words)
        #         union = pre_caption(ann['relation']['union'][0], self.max_words)
        
        image = Image.open(os.path.join(self.image_root,ann['image'])).convert('RGB') 
        w,h = image.size
        image = self.transform(image)

        gt_mask_indicator = 1.0
        
        return image, template, synonym, antonym, hypernym, meronym, obj_b, union, torch.LongTensor([gt_mask_indicator]), isSyn, isAnt, isHyp, isMer, isUni, caption
