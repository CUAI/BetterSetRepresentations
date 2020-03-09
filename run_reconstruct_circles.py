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
from models import *
import argparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', help='model type: srn | mlp', default="srn")
    parser.add_argument('--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('--lr', type=float, help='lr', default=3e-4)
    parser.add_argument('--inner_lr',type=float, help='inner lr', default=0.1)
    parser.add_argument('--inner_iters',type=int, help='# of inner iterations steps to perform', default=10)
    parser.add_argument('--start_epoch',type=int, help='epoch to start at', default=0)
    parser.add_argument('--load_ckpt', default=False, action='store_true')

    args = parser.parse_args()
    return args

class SSLR(nn.Module):
    def __init__(self, lr=200, num_iters=10, use_srn=True):
        super(SSLR, self).__init__()
        self.element_dims = 10
        self.set_generator = SetGen(element_dims = self.element_dims, set_size=16, lr=lr, use_srn=use_srn, iters=num_iters)
        self.f_reconstruct = F_reconstruct(element_dims = self.element_dims)
        self.use_srn = use_srn

    def forward(self, x, print_interm=False):
        x, losses = self.set_generator(x)
        generated_f, generated_set = self.f_reconstruct(x)

        if self.use_srn:
            return generated_f, losses, generated_set
        else:
            return generated_f, [], generated_set



def eval(net, batch_size, test_loader, epoch, writer, use_srn = True):
    net.eval()
    all_loss = 0
    rel_error = 0
    for idx, data in enumerate(test_loader):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()

        if use_srn:
            p, inner_losses, gs = net(images)
        else:
            p = net(images)

        loss = F.binary_cross_entropy(p, images)

        for j, s_ in enumerate(gs[0]):
            fig = plt.figure()
            plt.imshow(s_.transpose(0,2).detach().cpu())
            writer.add_figure(f"epoch-{epoch}/img-{idx}", fig, global_step=j)

        fig = plt.figure()
        plt.imshow(p[0].transpose(0,2).detach().cpu())
        writer.add_figure(f"epoch-{epoch}/img-{idx}", fig, global_step=len(gs[0]))

        fig = plt.figure()
        plt.imshow(images[0].transpose(0,2).detach().cpu())
        writer.add_figure(f"epoch-{epoch}/img-{idx}-target", fig, global_step=epoch)
        all_loss += loss.item()
    return all_loss/len(test_loader)

if __name__ == "__main__":
    args = get_args()
    print(args)
    use_srn = args.model_type == "srn"

    batch_size = args.batch_size
    train_loader = data.get_loader(data.MarkedColorCircles(train=True, size=64000), batch_size = batch_size)
    test_loader = data.get_loader(data.MarkedColorCircles(train=False, size=4000), batch_size = batch_size)

    use_srn = True
    net = SSLR(lr = args.inner_lr, num_iters=args.inner_iters, use_srn=use_srn).float().cuda()
    if args.load_ckpt:
        net.load_state_dict(torch.load("set_model_recon.pt"))


    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    writer = SummaryWriter(f"recon_run/test_run", purge_step=0, flush_secs = 10)

    print(type(net))
    print(net.set_generator.decoder.iters)
    running_loss = 0
    best_loss = 1e50
    for epoch in range(args.start_epoch, 1000+1):
        if epoch == 20:
            net.set_generator.decoder.iters = 20
        net.train()
        print(f"epoch {epoch}")
        running_loss = 0
        for idx, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()

            if use_srn:
                p, inner_losses, _ = net(images)
            else:
                p = net(images)
            loss = ((images - p)**2).sum()
            writer.add_scalar("train/loss", loss.item(), global_step=epoch*len(train_loader) + idx)

            loss.backward()
            optimizer.step()
            if idx % (len(train_loader)//4) == 0:
                if use_srn:
                    print(f"inner loss {[l.item()/batch_size for l in inner_losses]}")
                print(loss.item())
            running_loss += loss.item()

        print(running_loss/len(train_loader))
        if epoch % 1 ==0:
            eval_loss = eval(net, batch_size, test_loader, epoch, writer, use_srn)
            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(net.state_dict(), "set_model_recon.pt")
            print(f"eval: {eval_loss}")
            writer.add_scalar("eval/loss", eval_loss, global_step=epoch)
            writer.flush()

        print()
