from run_reconstruct_clevr import SSLR
import os
import data
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
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

    dataset_test = data.CLEVRMasked(
        "clevr", "test", full=True, iou=True
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

    SMOOTH = 1e-6
    full_iou = 0
    import gc
    for idx, data in enumerate(test_loader):
        def tfunc():
            gc.collect()
            image, image_mask, image_foreground_ = [x.cuda() for x in data]

            p_, inner_losses, gs_ = net(image)
            
            image, image_mask, image_foreground = [x.detach().cpu().numpy() for x in data]

            p = p_.detach().cpu().numpy()
            gs = gs_.detach().cpu().numpy()

            thresh_mask = p < 1e-2
            p[thresh_mask] = 0
            p[~thresh_mask] = 1
            p = p.astype('uint8')
            
            image_foreground[image_foreground != 0] = 1
            image_foreground = image_foreground.astype('uint8')
            
            intersect = (p & image_foreground).sum((1,2,3))
            union = (p | image_foreground).sum((1,2,3))
            iou = ((intersect + SMOOTH) / (union + SMOOTH)).sum()
            
            return iou
        full_iou += tfunc()
            
    full_iou /= len(test_loader) * batch_size
    print(full_iou)
