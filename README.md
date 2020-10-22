# [NeurIPS 2020] Better Set Representations For Relational Reasoning 

![main figure](https://github.com/CUAI/BetterSetRepresentations/blob/master/imgs/set_.pdf)

## Software Requirements

This codebase requires Python 3, PyTorch 1.0+, Torchvision 0.2+. In principle, this code can be run on CPU but we assume GPU utilization throughout the codebase.

## Usage

The files `run_reconstruct_circles.py`, `run_reconstruct_clevr.py` correspond with the explanatory experiments in the paper. We implemented the three other experiments by simply plugging our module into existing repos linked in supplementary materials, where we specify more details. 

Full usages:
```
usage: run_reconstruct_circles.py [-h] [--model_type MODEL_TYPE]
                                  [--batch_size BATCH_SIZE] [--lr LR]
                                  [--inner_lr INNER_LR]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        model type: srn | mlp
  --batch_size BATCH_SIZE
                        batch size
  --lr LR               lr
  --inner_lr INNER_LR   inner lr
```
```
usage: run_reconstruct_clevr.py [-h] [--model_type MODEL_TYPE]
                                [--batch_size BATCH_SIZE] [--lr LR]
                                [--inner_lr INNER_LR]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        model type: srn | mlp
  --batch_size BATCH_SIZE
                        batch size
  --lr LR               lr
  --inner_lr INNER_LR   inner lr
  --save SAVE           path to save checkpoint
  --resume RESUME       path to resume a saved checkpoint
```  

## Data Generation 

The data for CLEVR with masks was generated using https://github.com/facebookresearch/clevr-dataset-gen and adding the following line: 
```render_shadeless(blender_objects, path=output_image[:-4]+'_mask.png')```
on file ```image_generation/render_images.py``` ~line 311 (after the function ```add_random_objects``` is called).

## Results

Circles reconstruction samples (From left to right, column-wise: original images, SRN reconstruction, SRN decomposition, baseline reconstruction, baseline decomposition.):

![main figure](https://github.com/CUAI/BetterSetRepresentations/blob/master/imgs/tiled_samples_1.png)

CLEVR reconstruction samples:

![main figure](https://github.com/CUAI/BetterSetRepresentations/blob/master/imgs/clevr_tile_1.jpg)

