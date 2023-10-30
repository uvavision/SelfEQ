## Self-Consistency

### Requirements
- Python 3.8
- PyTorch 1.8.0+cu111
- transformers==4.8.1
- Numpy, scikit-image, opencv-python, pillow, matplotlib, timm

### Data
- Visual Genome (VG)
  - [images](https://visualgenome.org/).
  - [annotations](https://drive.google.com/drive/folders/1XhFVjJ2cm2HNeNVOZrUrPG_MpprHLWgv?usp=share_link).
- MS-COCO images
  - [images](https://cocodataset.org/#download).
  - [annotations 2014](https://cocodataset.org/#download).
- Our self-consistency augmented annotations. [download](https://drive.google.com/drive/folders/1k0eEor_hbUlwZZLw3E4VWaffMRRf04O0?usp=drive_link).

### Train
You can run the following command to train the model:
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env Pretrain_vg.py --config configs/Pretrain_vg.yaml --output_dir ALBEF_VG --checkpoint ALBEF.pth 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env Pretrain_coco.py --config configs/Pretrain_coco.yaml --output_dir ALBEF_COCO --checkpoint ALBEF.pth 
```

### Evaluation
To evaluate model performance on RefCOCO+, RefCLEF, and Flickr30k datasets, please run the following commands. ```--checkpoint``` supports a single checkpoint or all checkpoints under a directory.
```Shell
CUDA_VISIBLE_DEVICES=0 python grounding_eval_singlegpu_refclef.py --checkpoint ALBEF_VG --output_dir ALBEF_VG/refclef_results --config configs/Grounding_refclef.yaml
CUDA_VISIBLE_DEVICES=0 python grounding_eval_singlegpu_flickr.py --checkpoint ALBEF_VG --output_dir ALBEF_VG/flickr_results --config configs/Grounding_flickr.yaml
CUDA_VISIBLE_DEVICES=0 python grounding_eval_singlegpu.py --checkpoint ALBEF_VG --output_dir ALBEF_VG/refcoco_results --config configs/Grounding_refcoco.yaml
```

We provide our pretrained [checkpoints](https://drive.google.com/drive/folders/1k0eEor_hbUlwZZLw3E4VWaffMRRf04O0?usp=drive_link). To reproduce our results, please modify the checkpoint paths and run following commands for evaluation.
```Shell
CUDA_VISIBLE_DEVICES=0 python grounding_eval_singlegpu_refclef.py --checkpoint checkpoint_vg.pth --output_dir ALBEF_VG/refclef_results --config configs/Grounding_refclef.yaml
CUDA_VISIBLE_DEVICES=0 python grounding_eval_singlegpu_flickr.py --checkpoint checkpoint_vg.pth --output_dir ALBEF_VG/flickr_results --config configs/Grounding_flickr.yaml
CUDA_VISIBLE_DEVICES=0 python grounding_eval_singlegpu.py --checkpoint checkpoint_vg.pth --output_dir ALBEF_VG/refcoco_results --config configs/Grounding_refcoco.yaml
```
