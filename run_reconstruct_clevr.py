import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from dspn import DSPN
from fspool import FSPool
from tensorboardX import SummaryWriter
import matplotlib
import utils
from tqdm import tqdm
from models import *
import argparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', help='model type: srn | mlp', default="srn")
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--lr', type=float, help='lr', default=3e-4)
    parser.add_argument('--inner_lr',type=float, help='inner lr', default=8)
    parser.add_argument('--save', help='path to save checkpoint', default=None)
    parser.add_argument('--resume', help='path to resume a saved checkpoint', default=None)
    args = parser.parse_args()
    return args


class SSLR(nn.Module):
    def __init__(self, lr=8, use_srn=True):
        super(SSLR, self).__init__()
        self.use_srn = use_srn
        element_dims=10
        set_size=16
        self.g = SetGenCLEVR(element_dims, set_size, lr, use_srn)
        self.F_reconstruct = F_reconstruct_CLEVR()
    
    def forward(self, images):
        x, inner_losses = self.g(images)
        generated_f, generated_set = self.F_reconstruct(x)
        return generated_f, inner_losses, generated_set
        

def eval(net, batch_size, test_loader, epoch, writer, use_srn=True):
    with torch.no_grad():
        net.eval()
        all_loss = 0
        rel_error = 0
        test_loader = tqdm(
                test_loader,
                ncols=0,
                desc="test E{0:02d}".format(epoch),
            )
        iters_per_epoch = len(test_loader)
        for idx, (images, images_foreground) in enumerate(test_loader, start=epoch * iters_per_epoch):
            images, images_foreground = images.cuda(), images_foreground.cuda()
            
            p, inner_losses, gs = net(images)

            loss = F.binary_cross_entropy(p, images_foreground) 

            for j, s_ in enumerate(gs[0]):
                fig = plt.figure()
                plt.imshow(s_.permute(1,2,0).detach().cpu())
                writer.add_figure(f"epoch-{epoch}/img-{idx}", fig, global_step=j)
                
            fig = plt.figure()
            plt.imshow(p[0].permute(1,2,0).detach().cpu())
            writer.add_figure(f"epoch-{epoch}/img-{idx}", fig, global_step=len(gs[0]))
        
            fig = plt.figure()
            plt.imshow(images[0].permute(1,2,0).detach().cpu())
            writer.add_figure(f"epoch-{epoch}/img-{idx}-target", fig, global_step=epoch)

            all_loss += loss.item()
    return all_loss/len(test_loader)


    
    
if __name__ == "__main__":
    args = get_args()
    print(args)

    use_srn = args.model_type == "srn"
    
    dataset_train = data.CLEVRMasked(
        "clevr", "train", full=True
    )
    dataset_test = data.CLEVRMasked(
        "clevr", "test", full=False
    )

    batch_size = args.batch_size
    train_loader = data.get_loader(
        dataset_train, batch_size=batch_size
    )
    test_loader = data.get_loader(
        dataset_test, batch_size=batch_size
    )
    
    net = SSLR(args.inner_lr, use_srn).float().cuda()

    if args.resume:
        net.load_state_dict(torch.load(args.resume))

    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=args.lr
    )
    writer = SummaryWriter(f"runs/recon_clevr", purge_step=0, flush_secs = 10)
    
            
    print(type(net))
    iters_per_epoch = len(train_loader)

    running_loss = 0

    for epoch in range(1000+1):
        train_loader = tqdm(
            train_loader,
            ncols=0,
            desc="train E{0:02d}".format(epoch),
        )

        net.train()
        running_loss = 0

        for idx, (images, images_foreground) in enumerate(train_loader, start=epoch * iters_per_epoch):
            images, images_foreground = images.cuda(), images_foreground.cuda()
            optimizer.zero_grad()
            
            p, inner_losses, _ = net(images)

            loss = F.binary_cross_entropy(p, images_foreground)

            writer.add_scalar("train/loss", loss.item(), global_step=idx)

            loss.backward()
            optimizer.step()

            if use_srn:
                print(f"inner loss {[l.item()/batch_size for l in inner_losses]}")
            print(f"{loss.item()}\n")
            running_loss += loss.item() 
        print(running_loss/len(train_loader))

        if args.save:
            torch.save(net.state_dict(), args.save)

        eval_loss = eval(net, batch_size, test_loader, epoch, writer, use_srn)
        print(f"eval: {eval_loss}\n")
        writer.add_scalar("eval/loss", eval_loss, global_step=epoch)
        writer.flush()
            
        print()
        