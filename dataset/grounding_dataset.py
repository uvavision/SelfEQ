import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption
import numpy as np

class grounding_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, mode='train'):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.mode = mode
        
        if self.mode == 'train':
            self.img_ids = {} 
            n = 0
            for ann in self.ann:
                img_id = ann['image'].split('/')[-1]
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1            
        

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])            
        image = Image.open(image_path).convert('RGB')  
        image = self.transform(image)
        t = ann['text']
        caption = pre_caption(t, self.max_words) 
        
        if self.mode=='train':
            img_id = ann['image'].split('/')[-1]

            return image, caption, self.img_ids[img_id]
        elif 'split' in ann:
            return image, caption, ann['ref_id'], ann['image'], ann['split']
        else:
            return image, caption, ann['ref_id'], ann['image']
        
        
class grounding_dataset_flickr(Dataset):
    def __init__(self, transform, image_dir, image_ids_txt_file, phrase_boxes_json, sentences_json):
        super().__init__()
        self.max_words = 50
        
        self.transform = transform
        self.image_dir = image_dir
        self.splits = []
        self.processed_image_path = []
        self.processed_phrases = []
        self.processed_gt_boxes = []
        
        assert len(image_ids_txt_file)==len(phrase_boxes_json)==len(sentences_json)

        for i in range(0,len(image_ids_txt_file)):
            image_ids_txt = open(image_ids_txt_file[i], 'rb').read()
            image_ids = [idx.decode() for idx in image_ids_txt.split()]
            phrase_boxes = json.load(open(phrase_boxes_json[i],'r'))
            sentences = json.load(open(sentences_json[i],'r'))
        
            for img_id in image_ids:
                sents = sentences[img_id]
                p_boxes = phrase_boxes[img_id]
                for sent in sents:
                    for phrase in sent['phrases']:
                        if phrase['phrase_id'] not in phrase_boxes[img_id]['boxes']:
                            continue
                        else:
                            if 'val' in image_ids_txt_file[i]:
                                self.splits.append('val')
                            else:
                                self.splits.append('test')
                            self.processed_image_path.append(img_id)
                            self.processed_phrases.append(pre_caption(phrase['phrase'],self.max_words))
                            self.processed_gt_boxes.append(phrase_boxes[img_id]['boxes'][phrase['phrase_id']])
                    
        

    def get_image_path(self,image_id):
        return os.path.join(
            self.image_dir,
            f'{image_id}.jpg')
            
    def __len__(self):
        return len(self.processed_image_path)

    def __getitem__(self,i):

        image_id = self.processed_image_path[i]
        phrase = self.processed_phrases[i]
        gt_boxes = self.processed_gt_boxes[i]
        
        image_path = os.path.join(self.image_dir,image_id+'.jpg')
        image_pil = Image.open(image_path).convert('RGB') 
        w,h = image_pil.size
        image = self.transform(image = np.array(image_pil))['image']
        
        return image, phrase, self.splits[i], h, w, i