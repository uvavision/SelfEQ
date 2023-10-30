import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.model_eval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from dataset.utils import collect_result, grounding_eval_flickr
from scheduler import create_scheduler
from optim import create_optimizer


from pdb import set_trace as breakpoint

from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

def val(model, data_loader, tokenizer, device, gradcam_mode, block_num):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 150
    
    if gradcam_mode=='itm':
        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True
     
    result = []
    
    for image, text, splits, heights, widths, data_index in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device)
        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)  
        

        image_embeds = model.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        output = model.text_encoder(text_input.input_ids, 
                                attention_mask = text_input.attention_mask,
                                encoder_hidden_states = image_embeds,
                                encoder_attention_mask = image_atts,      
                                return_dict = True,
                               )     

        vl_embeddings = output.last_hidden_state[:,0,:]
        vl_output = model.itm_head(vl_embeddings)   
        loss = vl_output[:,1].sum()   

        model.zero_grad()
        loss.backward()    

        with torch.no_grad():                                 
            mask = text_input.attention_mask.view(text_input.attention_mask.size(0),1,-1,1,1)

            grads = model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients().detach()
            cams = model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map().detach()

            cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, 16, 16) * mask              
            grads = grads[:, :, :, 1:].clamp(min=0).reshape(image.size(0), 12, -1, 16, 16) * mask

            gradcam = cams * grads
            gradcam = gradcam.mean(1).mean(1)


        for r_id, cam , split, hh, ww in zip(data_index, gradcam, splits, heights, widths):
            result.append({'ref_id':r_id.item(), 'pred':cam, 'image_height': hh.item(), 'image_width':ww.item(), 'split':split})
  
    if gradcam_mode=='itm':
        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = False             
    return result


def main(args, config):
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset #### 
    print("Creating dataset")
    grd_test_dataset = create_dataset('grounding_flickr', config) 
    datasets = [grd_test_dataset]
    samplers = [None, None]
    test_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], \
                                num_workers=[4], is_trains=[False], collate_fns=[None])[0]

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    
    model = ALBEF(config = config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    model = model.to(device)   
    
    if os.path.isdir(args.checkpoint):
        all_ = glob.glob('{}/*.pth'.format(args.checkpoint))
        all_.sort()

        for checkpoint in all_:
            filename = 'epoch'+checkpoint[-6:-4]

            
            final_result_file = os.path.join(args.result_dir, '%s.pth'%filename)
            if os.path.isfile(final_result_file):
                continue
                
            
            # load pre-trained model
            ckpt = torch.load(checkpoint, map_location='cpu') 

            state_dict = ckpt['model']
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

            for key in list(state_dict.keys()):
                if 'bert' in key:
                    encoder_key = key.replace('bert.','')         
                    state_dict[encoder_key] = state_dict[key] 
                    del state_dict[key]     
            msg = model.load_state_dict(state_dict,strict=False) 
            

            print('load checkpoint from %s'%checkpoint)
            print(msg)          
            del ckpt
            
            result = val(model, test_loader, tokenizer, device, args.gradcam_mode, args.block_num)
            torch.save(result,final_result_file)     
            print('result file saved to %s'%final_result_file)
                
            grounding_acc = grounding_eval_flickr(result,grd_test_dataset.processed_gt_boxes, mask_size=16)
            print(grounding_acc)
            log_stats = {**{f'{k}': v for k, v in grounding_acc.items()},
                             'epoch': checkpoint[-6:-4],
                            } 
            with open(os.path.join(args.result_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")  
    else:
        filename = 'epoch00'
        final_result_file = os.path.join(args.result_dir, '%s.pth'%filename)
        if os.path.isfile(final_result_file):
            return
        # load pre-trained model
        ckpt = torch.load(args.checkpoint, map_location='cpu') 

        state_dict = ckpt['model']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]     
        msg = model.load_state_dict(state_dict,strict=False) 

        print('load checkpoint from %s'%args.checkpoint)
        print(msg)          
        del ckpt

        result = val(model, test_loader, tokenizer, device, args.gradcam_mode, args.block_num)
        torch.save(result,final_result_file)     
        print('result file saved to %s'%final_result_file)
        
        grounding_acc = grounding_eval_flickr(result,grd_test_dataset.processed_gt_boxes, mask_size=16)
        log_stats = {**{f'{k}': v for k, v in grounding_acc.items()},
                         'epoch': '00',
                        } 
        with open(os.path.join(args.result_dir, "log.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Grounding_flickr.yaml')
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--output_dir', default='output/flickr')   
    parser.add_argument('--gradcam_mode', default='itm', choices=['itm','itc']) 
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
