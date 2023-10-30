'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_self_consistency import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
            
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, dt in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image, text_input, mask_query_interp, gt_mask_indicator = dt
        text_input = tokenizer(text_input, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)
        
        if not config['add_gcam']:
            mask_query_interp = None
            gt_mask_indicator = None

        optimizer.zero_grad()        
        image = image.to(device,non_blocking=True) 
      
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader)) 
        
        loss_mlm, loss_ita, loss_itm = model(image=image,text=text_input, alpha = alpha, mask_query_interp = mask_query_interp, gt_mask_indicator = gt_mask_indicator)  
        
        loss = loss_mlm + loss_ita + loss_itm    
          
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
            
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])   
       
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    
    

def relation(model, relation_loader, relation_optimizer, tokenizer, epoch, warmup_steps, device, relation_lr_scheduler, config):
    # relation
    model.train()  
    
    relation_logger = utils.MetricLogger(delimiter="  ")
    relation_logger.add_meter('relation_lr', utils.SmoothedValue(window_size=50, fmt='{value:.7f}'))
    relation_logger.add_meter('loss_relation', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    relation_logger.add_meter('loss_syn', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    relation_logger.add_meter('loss_syn_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    relation_logger.add_meter('loss_syn_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    relation_logger.add_meter('loss_syn_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    relation_logger.add_meter('loss_sim_syn', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    relation_logger.add_meter('loss_cst_syn', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

            
    header = 'Relation Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    if args.distributed:
        relation_loader.sampler.set_epoch(epoch)

    for i, dt in enumerate(relation_logger.log_every(relation_loader, print_freq, header)):

        image, template, synonym, antonym, hypernym, meronym, obj_b, union, gt_mask_indicator, isSyn, isAnt, isHyp, isMer, isUni, caption = dt
        synonym = [s for i, s in enumerate(synonym) if isSyn[i]]
        synonym = tokenizer(synonym, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)
        antonym = [a for i, a in enumerate(antonym) if isAnt[i]]
        antonym = tokenizer(antonym, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)

        template = tokenizer(template, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)
        caption = tokenizer(caption, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)

        relation_optimizer.zero_grad()        
        image = image.to(device,non_blocking=True) 
      
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(relation_loader)) 
        
        loss, relation_stats = model(image=image, template=template, synonym=synonym, antonym=antonym, hypernym=hypernym, meronym=meronym, alpha=alpha, gt_mask_indicator=gt_mask_indicator, isSyn=isSyn, isAnt=isAnt, isHyp=isHyp, isMer=isMer, isUni=isUni, text=caption, isRelation=True)  
  
        loss.backward()
        relation_optimizer.step()    
            
        relation_logger.update(relation_lr=relation_optimizer.param_groups[0]["lr"])  
        relation_logger.update(loss_relation=loss.item()) 
        relation_logger.update(loss_syn=relation_stats['loss_syn'].item()) 
        relation_logger.update(loss_syn_mlm=relation_stats['loss_syn_mlm'].item()) 
        relation_logger.update(loss_syn_ita=relation_stats['loss_syn_ita'].item()) 
        relation_logger.update(loss_syn_itm=relation_stats['loss_syn_itm'].item()) 
        relation_logger.update(loss_sim_syn=relation_stats['loss_sim_syn'].item()) 
        relation_logger.update(loss_cst_syn=relation_stats['loss_cst_syn'].item()) 
       
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            relation_lr_scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    relation_logger.synchronize_between_processes()
    print("Averaged stats:", relation_logger.global_avg())     
    return {k: "{:.7f}".format(meter.global_avg) for k, meter in relation_logger.meters.items()}  


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']    

    #### Dataset #### 
    print("Creating dataset")
    datasets = [create_dataset('pretrain', config)]
    relation_dataset = [create_dataset('relation_1', config)]
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)     
        relation_samplers = create_sampler(relation_dataset, [True], num_tasks, global_rank)    
    else:
        samplers = [None]
        relation_samplers = [None]

    data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
    relation_loader = create_loader(relation_dataset, relation_samplers, batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    
    model = model.to(device)   
    if config['add_gcam']:
        model.text_encoder.base_model.base_model.encoder.layer[8].crossattention.self.save_attention = True
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer) 

    arg_relationopt = utils.AttrDict(config['relation_optimizer']) 
    relation_optimizer = create_optimizer(arg_relationopt, model)
    arg_relationsche = utils.AttrDict(config['relation_schedular'])
    relation_lr_scheduler, _  = create_scheduler(arg_relationsche, relation_optimizer)

    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                       
        if args.resume:
            optimizer.load_state_dict(checkpoint['relation_oprimizer'])
            lr_scheduler.load_state_dict(checkpoint['relation_lr_scheduler'])
            start_epoch = checkpoint['epoch']+1         
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)  
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped               
        msg = model.load_state_dict(state_dict, strict=False)    
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]) #, find_unused_parameters=True
        model_without_ddp = model.module    
    
    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, 1):
            
        if epoch>0: 
            relation_lr_scheduler.step(epoch+warmup_steps)
            
        relation_stats = relation(model, relation_loader, relation_optimizer, tokenizer, epoch, warmup_steps, device, relation_lr_scheduler, config)
        if utils.is_main_process():
            log_stats = {**{f'relation_{k}': v for k, v in relation_stats.items()},
                         'epoch': epoch,
                        }                     
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'relation_oprimizer': relation_optimizer.state_dict(),
                'relation_lr_scheduler': relation_lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_relation_%02d.pth'%epoch))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()  

    for epoch in range(1, max_epoch):
            
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
            
        train_stats = train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config) 

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()  
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)