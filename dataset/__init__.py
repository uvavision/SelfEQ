import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import pretrain_dataset
from dataset.grounding_dataset import grounding_dataset, grounding_dataset_flickr
from dataset.relation_dataset_v1 import relation_dataset_1
from dataset.relation_dataset_coco import relation_dataset_coco
from dataset.coco_dataset import pretrain_dataset_coco
from dataset.randaugment_imgonly import RandomAugment_img
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
from dataset.moco.augmentations import transforms as T
import cv2
def create_dataset(dataset, config):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    alb_transform = alb.Compose([
                alb.Resize(config['image_res'],config['image_res']),
                alb.HorizontalFlip(p=0.5),
                alb.ToGray(p=0.2),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=0.5),
                alb.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), p=1.0),
                ToTensorV2(),
            ],bbox_params=alb.BboxParams(format='pascal_voc'))
    
    weak_transform = alb.Compose([
                alb.Resize(config['image_res'],config['image_res']),
                alb.HorizontalFlip(p=0.5),
                alb.ToGray(p=0.2),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=0.5),
                alb.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), p=1.0),
                ToTensorV2(),
            ])

    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment_img(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
 
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    if dataset=='pretrain':
        if config['add_gcam']:     
            t = [alb_transform]
            dataset = pretrain_dataset(config['train_file'], t, config['image_root'], 
                                       config['max_words'], config['phrase_input'], config['add_gcam'], config['mask_all'],
                                       config['mask_size'])
        else:
            dataset = pretrain_dataset(config['train_file'], pretrain_transform, config['image_root'],
                                       config['max_words'], config['phrase_input'])
        return dataset  
    
    if dataset=='pretrain_coco':
        dataset = pretrain_dataset_coco(config['train_file'], weak_transform, config['image_root'],
                                       config['max_words'], config['phrase_input'])
        return dataset  

    elif dataset=='relation_1':
        relation = relation_dataset_1(config['relation_file'], test_transform, config['image_root'], config['max_words'], config['phrase_input']) 
        return relation
    
    elif dataset=='relation_coco':
        relation = relation_dataset_coco(config['relation_file'], test_transform, config['image_root'], config['max_words'], config['phrase_input']) 
        return relation

    elif dataset=='grounding':   
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')             
        return test_dataset   
    
    elif dataset == 'grounding_flickr':
        test_flickr_transform = alb.Compose([
            alb.Resize(256,256),
            alb.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), p=1.0),
            ToTensorV2(),
        ])
        test_dataset = grounding_dataset_flickr(test_flickr_transform, config['image_root'], config['image_ids_file'], config['phrase_box_file'], config['sentences_file'])     
        return test_dataset 

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    