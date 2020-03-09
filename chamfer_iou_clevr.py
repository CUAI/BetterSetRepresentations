from run_reconstruct_clevr import SSLR
import os
import data
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils import chamfer_score, cv_bbox
import torch.multiprocessing as mp
import gc
import cv2
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', help='model type: srn | mlp', default="srn")
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--resume', help='path to resume a saved checkpoint', default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(args)

    use_srn = args.model_type == "srn"
    
    dataset_test = data.CLEVR(
        "clevr_no_mask", "val", box=True, full=True, chamfer=True
    )
    batch_size = args.batch_size
    test_loader = data.get_loader(
        dataset_test, batch_size=batch_size, shuffle=False
    )

    net = SSLR(use_srn=use_srn).float().cuda()
    net.eval()
    net.load_state_dict(torch.load(args.resume))

    test_loader = tqdm(
            test_loader,
            ncols=0,
            desc="test"
        )

    full_score = 0
    for idx, sample in enumerate(test_loader):
        def tfunc():
            gc.collect()
            image, masks = [x.cuda() for x in sample]

            p_, inner_losses, gs = net(image)

            thresh_mask = gs < 1e-2
            gs[thresh_mask] = 0
            gs[~thresh_mask] = 1
            gs = gs.sum(2).clamp(0,1)
            gs = gs.to(dtype=torch.uint8)

            img = cv_bbox(gs.detach().cpu().numpy().reshape(-1,128,128))

            score = chamfer_score(img.cuda().to(dtype=torch.uint8), masks.to(dtype=torch.uint8))

            return score
        full_score += tfunc()

            
    full_score /= len(test_loader)
    print(full_score)
