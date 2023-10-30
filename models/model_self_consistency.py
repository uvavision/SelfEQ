'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random


class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        self.itm_head = nn.Linear(text_width, 2)
        

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)       
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        self.loss_sim = nn.MSELoss(reduction='sum')
        self.loss_bce = nn.BCELoss()
        self.loss_smoothl1 = nn.SmoothL1Loss()
        
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr_img", torch.zeros(1, dtype=torch.long))  
        self.register_buffer("queue_ptr_txt", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.mask_size = config['mask_size']
        self.add_gcam = config['add_gcam']

    def forward_train(self, image, text, alpha=0, mask_query_interp = None, gt_mask_indicator = None):
        '''training'''
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  

        text_output = self.text_encoder.bert(text['input_ids'], \
                                             attention_mask = text['attention_mask'],\
                                             return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image) 
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)  
            
                                                   
            text_output_m = self.text_encoder_m.bert(text['input_ids'], \
                                                     attention_mask = text['attention_mask'],                      
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)  

            sim_targets_t2i = torch.zeros(sim_t2i_m.size()).to(image.device)
            sim_targets_t2i.fill_diagonal_(1)  

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets_t2i        

        sim_i2t = image_feat @ text_feat_all / self.temp 
        sim_t2i = text_feat @ image_feat_all / self.temp 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2
        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                        attention_mask = text['attention_mask'],
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )    
        # if mask_query_interp is not None and self.add_gcam:
        #     fmaps=self.text_encoder.base_model.base_model.encoder.layer[8].crossattention.self.get_attention_map()
        with torch.no_grad():
            bs = image.size(0)          
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)
   
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text['attention_mask'][neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text['attention_mask'], text_atts_neg],dim=0)         
        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all, 
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                         

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)  
        

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)     
            
        ##================= MLM ========================##                
        input_ids = text['input_ids'].clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids, 
                                           attention_mask = text['attention_mask'],
                                           encoder_hidden_states = image_embeds_m,
                                           encoder_attention_mask = image_atts,      
                                           return_dict = True,
                                           return_logits = True,   
                                          )    
        mlm_output = self.text_encoder(input_ids, 
                                       attention_mask = text['attention_mask'],
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       labels = labels,   
                                       soft_labels = F.softmax(logits_m,dim=-1),
                                       alpha = alpha
                                      )                           
        loss_mlm = mlm_output.loss        

        return loss_mlm, loss_ita, loss_itm  

    def forward(self, image, text=None, template=None, synonym=None, antonym=None, hypernym=None, meronym=None, obj_b=None, union=None, alpha=0, mask_query_interp = None, gt_mask_indicator = None, isSyn = None, isAnt = None, isHyp = None, isMer = None, isUni = None, isRelation=False):
        if not isRelation:
            return self.forward_train(image=image,text=text, alpha=alpha, mask_query_interp=mask_query_interp, gt_mask_indicator=gt_mask_indicator)
        else:
            return self.forward_relation(image=image, template=template, synonym=synonym, alpha = alpha, gt_mask_indicator=gt_mask_indicator, isSyn=isSyn, isAnt=isAnt, isHyp=isHyp, isMer=isMer, isUni=isUni, text=text)

    def forward_relation(self, image, text, template, synonym, alpha=0, mask_query_interp = None, gt_mask_indicator = None, isSyn=None, isAnt=None, isHyp=None, isMer=None, isUni=None):
        '''relation'''
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        loss_relation = torch.tensor(0.0).to(image.device)
        loss_syn = torch.tensor(0.0).to(image.device)
        loss_ant = torch.tensor(0.0).to(image.device)
        loss_hyp = torch.tensor(0.0).to(image.device)
        loss_mer = torch.tensor(0.0).to(image.device)
        loss_uni = torch.tensor(0.0).to(image.device)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  

        text_output = self.text_encoder.bert(text['input_ids'], \
                                             attention_mask = text['attention_mask'],\
                                             return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)

        ################## synonym ##################
        isSyn_mask = isSyn.unsqueeze(1).unsqueeze(2)
        synonym_image_embeds = image_embeds[isSyn_mask.expand_as(image_embeds)].view(-1, image_embeds.size(1), image_embeds.size(2))
        synonym_image_atts = torch.ones(synonym_image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        synonym_image_feat = F.normalize(self.vision_proj(synonym_image_embeds[:,0,:]),dim=-1)  
        synonym_output = self.text_encoder.bert(synonym['input_ids'], \
                                            attention_mask = synonym['attention_mask'],\
                                            return_dict = True, mode = 'text')        
        synonym_embeds = synonym_output.last_hidden_state
        synonym_feat = F.normalize(self.text_proj(synonym_embeds[:,0,:]),dim=-1)
        
        # get momentum features
        with torch.no_grad():
            self._momentum_update()           
            # get momentum features  
            image_embeds_m = self.visual_encoder_m(image)                            
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                                         
            text_output_m = self.text_encoder_m.bert(text['input_ids'], \
                                                     attention_mask = text['attention_mask'],                      
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)  

            sim_targets_t2i = torch.zeros(sim_t2i_m.size()).to(image.device)
            sim_targets_t2i.fill_diagonal_(1)  

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets_t2i  


            ############## Synonym ###############
            isSyn_mask = isSyn.unsqueeze(1).unsqueeze(2)  # Expand dimensions to match image_embeds_m shape
            # Use the mask to filter the image_embeds_m
            synonym_image_embeds_m =  image_embeds_m[isSyn_mask.expand_as(image_embeds_m)].view(-1, image_embeds_m.size(1), image_embeds_m.size(2))
            
            synonym_image_feat_m = F.normalize(self.vision_proj_m(synonym_image_embeds_m[:,0,:]),dim=-1)  
            synonym_image_feat_all = torch.cat([synonym_image_feat_m.t(),self.image_queue.clone().detach()],dim=1) 
                 
            synonym_output_m = self.text_encoder_m.bert(synonym['input_ids'], \
                                                    attention_mask = synonym['attention_mask'],                      
                                                    return_dict = True, mode = 'text')    
            synonym_feat_m = F.normalize(self.text_proj_m(synonym_output_m.last_hidden_state[:,0,:]),dim=-1) 
            
            synonym_feat_all = torch.cat([synonym_feat_m.t(),self.text_queue.clone().detach()],dim=1)
            
            synonym_sim_i2t_m = synonym_image_feat_m @ synonym_feat_all / self.temp 
            synonym_sim_t2i_m = synonym_feat_m @ synonym_image_feat_all / self.temp     

            synonym_sim_targets = torch.zeros(synonym_sim_i2t_m.size()).to(image.device)
            synonym_sim_targets.fill_diagonal_(1)  

            synonym_sim_targets_t2i = torch.zeros(synonym_sim_t2i_m.size()).to(image.device)
            synonym_sim_targets_t2i.fill_diagonal_(1)  

            synonym_sim_i2t_targets = alpha * F.softmax(synonym_sim_i2t_m, dim=1) + (1 - alpha) * synonym_sim_targets
            synonym_sim_t2i_targets = alpha * F.softmax(synonym_sim_t2i_m, dim=1) + (1 - alpha) * synonym_sim_targets_t2i         

        sim_i2t = image_feat @ text_feat_all / self.temp 
        sim_t2i = text_feat @ image_feat_all / self.temp 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2

        synonym_sim_i2t = synonym_image_feat @ synonym_feat_all / self.temp 
        synonym_sim_t2i = synonym_feat @ synonym_image_feat_all / self.temp 
                            
        loss_i2t_synonym = -torch.sum(F.log_softmax(synonym_sim_i2t, dim=1)*synonym_sim_i2t_targets,dim=1).mean()
        loss_t2i_synonym = -torch.sum(F.log_softmax(synonym_sim_t2i, dim=1)*synonym_sim_t2i_targets,dim=1).mean() 

        loss_ita_synonym = (loss_i2t_synonym+loss_t2i_synonym)/2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)
        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                        attention_mask = text['attention_mask'],
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )
        
        # if mask_query_interp is not None and self.add_gcam:
        fmaps=self.text_encoder.base_model.base_model.encoder.layer[8].crossattention.self.get_attention_map()
        with torch.no_grad():
            bs = image.size(0)          
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)
   
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            #print(weights_t2i[b]) nan error control
            weights_t2i[b] = torch.where(torch.isnan(weights_t2i[b]), torch.tensor(1e-4, device=weights_t2i.device), weights_t2i[b])
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            weights_i2t[b] = torch.where(torch.isnan(weights_i2t[b]), torch.tensor(1e-4, device=weights_i2t.device), weights_i2t[b])
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text['attention_mask'][neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text['attention_mask'], text_atts_neg],dim=0)         
        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all, 
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                         

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)  
        

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)  

        gt_mask_indicator = gt_mask_indicator.squeeze()
        
        mask = text['attention_mask'].view(bs,1,-1,1,1)
        output = vl_output[:bs]
        one_hot = torch.zeros_like(output)
        one_hot[:,1]=1
        
        grad_wrt_act = torch.autograd.grad(outputs=output, inputs=fmaps, grad_outputs=one_hot, \
                                            create_graph=True)[0]        

        fmaps = fmaps[:, :, :, 1:].reshape(bs, 12, -1, 16, 16) * mask
        
        
        grad_wrt_act = grad_wrt_act[:, :, :, 1:].clamp(0).reshape(bs, 12, -1, 16, 16) * mask
        
        gradcam = fmaps * grad_wrt_act
        gradcam = gradcam[gt_mask_indicator>0]

        gradcam = gradcam.mean(1).mean(1)
        B, H, W = gradcam.shape

        gradcam = F.relu(gradcam, inplace = False)

        gradcam = gradcam.view(B, -1)
        gradcam -= gradcam.min(dim=1, keepdim=True)[0]
        gradcam /= (gradcam.max(dim=1, keepdim=True)[0]+1e-7)
        # gradcam = gradcam.view(B, H, W)
        
        # # scale gradcam to image size
        # gradcam = gradcam.view(B,1,H,W)
        # gradcam = F.interpolate(gradcam, (self.mask_size, self.mask_size), mode="bilinear", align_corners=False).squeeze()

        ################## synonym ##################
        # forward the positve image-text pair
        output_pos_synonym = self.text_encoder.bert(encoder_embeds = synonym_embeds, 
                                        attention_mask = synonym['attention_mask'],
                                        encoder_hidden_states = synonym_image_embeds,
                                        encoder_attention_mask = synonym_image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                    )       

        fmaps_synonym=self.text_encoder.base_model.base_model.encoder.layer[8].crossattention.self.get_attention_map()
        with torch.no_grad():
            synonym_bs = synonym_image_embeds.size(0)          
            synonym_weights_i2t = F.softmax(synonym_sim_i2t[:,:synonym_bs],dim=1)
            synonym_weights_t2i = F.softmax(synonym_sim_t2i[:,:synonym_bs],dim=1)
   
            synonym_weights_i2t.fill_diagonal_(0)
            synonym_weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        synonym_image_embeds_neg = []    
        for b in range(synonym_bs):
            synonym_weights_t2i[b] = torch.where(torch.isnan(synonym_weights_t2i[b]), torch.tensor(1e-4, device=synonym_weights_t2i.device), synonym_weights_t2i[b])
            synonym_neg_idx = torch.multinomial(synonym_weights_t2i[b], 1).item()
            synonym_image_embeds_neg.append(synonym_image_embeds[synonym_neg_idx])
        synonym_image_embeds_neg = torch.stack(synonym_image_embeds_neg,dim=0)   

        # select a negative text for each image
        synonym_text_embeds_neg = []
        synonym_text_atts_neg = []
        for b in range(synonym_bs):
            synonym_weights_i2t[b] = torch.where(torch.isnan(synonym_weights_i2t[b]), torch.tensor(1e-4, device=synonym_weights_i2t.device), synonym_weights_i2t[b])
            synonym_neg_idx = torch.multinomial(synonym_weights_i2t[b], 1).item()
            synonym_text_embeds_neg.append(synonym_embeds[synonym_neg_idx])
            synonym_text_atts_neg.append(synonym['attention_mask'][synonym_neg_idx])

        synonym_text_embeds_neg = torch.stack(synonym_text_embeds_neg,dim=0)   
        synonym_text_atts_neg = torch.stack(synonym_text_atts_neg,dim=0)      

        synonym_text_embeds_all = torch.cat([synonym_embeds, synonym_text_embeds_neg],dim=0)     
        synonym_text_atts_all = torch.cat([synonym['attention_mask'], synonym_text_atts_neg],dim=0)         
        synonym_image_embeds_all = torch.cat([synonym_image_embeds_neg,synonym_image_embeds],dim=0)
        synonym_image_atts_all = torch.cat([synonym_image_atts,synonym_image_atts],dim=0)

        synonym_output_neg = self.text_encoder.bert(encoder_embeds = synonym_text_embeds_all, 
                                        attention_mask = synonym_text_atts_all,
                                        encoder_hidden_states = synonym_image_embeds_all,
                                        encoder_attention_mask = synonym_image_atts_all,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                         

        synonym_vl_embeddings = torch.cat([output_pos_synonym.last_hidden_state[:,0,:],synonym_output_neg.last_hidden_state[:,0,:]],dim=0)
        synonym_vl_output = self.itm_head(synonym_vl_embeddings)  
        

        synonym_itm_labels = torch.cat([torch.ones(synonym_bs,dtype=torch.long),torch.zeros(2*synonym_bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm_synonym = F.cross_entropy(synonym_vl_output, synonym_itm_labels)    

        ##================= MLM ========================##                
        synonym_input_ids = synonym['input_ids'].clone()
        synonym_labels = synonym_input_ids.clone()

        synonym_probability_matrix = torch.full(synonym_labels.shape, self.mlm_probability)                    
        synonym_input_ids, synonym_labels = self.mask(synonym_input_ids, self.text_encoder.config.vocab_size, image.device, targets=synonym_labels,
                                      probability_matrix = synonym_probability_matrix) 
        
        with torch.no_grad():
            synonym_logits_m = self.text_encoder_m(synonym_input_ids, 
                                           attention_mask = synonym['attention_mask'],
                                           encoder_hidden_states = synonym_image_embeds_m,
                                           encoder_attention_mask = synonym_image_atts,      
                                           return_dict = True,
                                           return_logits = True,   
                                          )    
        synonym_mlm_output = self.text_encoder(synonym_input_ids, 
                                       attention_mask = synonym['attention_mask'],
                                       encoder_hidden_states = synonym_image_embeds,
                                       encoder_attention_mask = synonym_image_atts,      
                                       return_dict = True,
                                       labels = synonym_labels,   
                                       soft_labels = F.softmax(synonym_logits_m,dim=-1),
                                       alpha = alpha
                                      )                           
        loss_mlm_synonym = synonym_mlm_output.loss         

        gt_mask_indicator = gt_mask_indicator.squeeze()

        ################### synonym gradcam ###################
        mask_synonym = synonym['attention_mask'].view(synonym_bs,1,-1,1,1)
        output_synonym = synonym_vl_output[:synonym_bs]
        one_hot_synonym = torch.zeros_like(output_synonym)
        one_hot_synonym[:,1]=1
        
        grad_wrt_act_synonym = torch.autograd.grad(outputs=output_synonym, inputs=fmaps_synonym, grad_outputs=one_hot_synonym, create_graph=True)[0]        

        fmaps_synonym = fmaps_synonym[:, :, :, 1:].reshape(synonym_bs, 12, -1, 16, 16) * mask_synonym
        
        grad_wrt_act_synonym = grad_wrt_act_synonym[:, :, :, 1:].clamp(0).reshape(synonym_bs, 12, -1, 16, 16) * mask_synonym
        
        gradcam_synonym = fmaps_synonym * grad_wrt_act_synonym  
        syn_gt_mask_indicator = gt_mask_indicator[isSyn]
        gradcam_synonym = gradcam_synonym[syn_gt_mask_indicator>0]

        gradcam_synonym = gradcam_synonym.mean(1).mean(1)
        B, H, W = gradcam_synonym.shape

        gradcam_synonym = F.relu(gradcam_synonym, inplace = False)

        gradcam_synonym = gradcam_synonym.view(B, -1)
        gradcam_synonym -= gradcam_synonym.min(dim=1, keepdim=True)[0]
        gradcam_synonym /= (gradcam_synonym.max(dim=1, keepdim=True)[0]+1e-7)

        isSyn_mask = isSyn.unsqueeze(1)
        gradcam4isSyn = gradcam[isSyn_mask.expand_as(gradcam)].view(-1, gradcam.size(1))
        gradcam_syn_text = gradcam4isSyn + gradcam_synonym
        t_syn_roi = torch.tensor(0.8).to(image.device)
        syn_roi_mask = (gradcam_syn_text >= t_syn_roi)
        syn_roi = torch.mul(gradcam_synonym, syn_roi_mask)
        text_roi = torch.mul(gradcam4isSyn, syn_roi_mask)
         
        loss_sim_syn = self.loss_smoothl1(gradcam_synonym, gradcam4isSyn)
        loss_consistency_syn = torch.std(syn_roi) + torch.std(text_roi) \
                                + torch.max(torch.tensor(0.).to(image.device), t_syn_roi/2 - syn_roi.sum()/syn_roi_mask.sum()) \
                                + torch.max(torch.tensor(0.).to(image.device), t_syn_roi/2 - text_roi.sum()/syn_roi_mask.sum())

        loss_syn = loss_mlm_synonym + loss_ita_synonym + loss_itm_synonym + loss_sim_syn + 0.5*loss_consistency_syn 
        
        loss_relation += loss_syn
        
        relation_stats = {
            'loss_syn': loss_syn,
            'loss_syn_mlm': loss_mlm_synonym,
            'loss_syn_ita': loss_ita_synonym,
            'loss_syn_itm': loss_itm_synonym,
            'loss_sim_syn': loss_sim_syn,
            'loss_cst_syn': loss_consistency_syn
        }

        return loss_relation, relation_stats

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        restart_img = False
        restart_text = False
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size_img = image_feats.shape[0]
        batch_size_text = text_feats.shape[0]

        ptr_img = int(self.queue_ptr_img)
        ptr_txt = int(self.queue_ptr_txt)

        # replace the keys at ptr (dequeue and enqueue)
        if ptr_img + batch_size_img < self.queue_size:
            self.image_queue[:, ptr_img:ptr_img + batch_size_img] = image_feats.T
        else:
            restart_img = True
            diff_img = self.queue_size - ptr_img 
            self.image_queue[:, ptr_img:] = image_feats.T[:,:diff_img]
        if ptr_txt + batch_size_text < self.queue_size:
            self.text_queue[:, ptr_txt:ptr_txt + batch_size_text] = text_feats.T
        else:
            restart_text = True
            diff_txt = self.queue_size - ptr_txt 
            self.text_queue[:, ptr_txt:] = text_feats.T[:,:diff_txt]
        ptr_img = (ptr_img + batch_size_img) % self.queue_size  # move pointer
        ptr_txt = (ptr_txt + batch_size_text) % self.queue_size  # move pointer

        if restart_img:
            self.image_queue[:,:ptr_img] = image_feats.T[:,diff_img:]
        if restart_text:
            self.text_queue[:, :ptr_txt] = text_feats.T[:,diff_txt:]

    
        self.queue_ptr_img[0] = ptr_img 
        self.queue_ptr_txt[0] = ptr_txt 
        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output