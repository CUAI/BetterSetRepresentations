import argparse
import data
import torch
import torch.nn as nn
from models import SetGen, F_match, F_reconstruct
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', help='model type: srn | mlp | cnn', default="srn")
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--recon', action="store_true" , help='transfer models', default=False)
    parser.add_argument('--resume', help='Resume checkpoint', default=None)
    parser.add_argument('--lr', type=float, help='lr', default=5e-4)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=0)
    parser.add_argument('--inner_lr',type=float, help='inner lr', default=0.1)
    parser.add_argument('--save', help='Path of the saved checkpoint', default=None)
    args = parser.parse_args()
    return args


class Net(nn.Module):
    def __init__(self, lr=200):
        super(Net, self).__init__()
        self.img_encoder = Encoder()
        self.proj = nn.Linear(100, 1)

    def forward(self, x, y):
        all_images = torch.cat((x, y))
        x, s = self.img_encoder(all_images)
        batch_size = x.size(0) // 2

        reference = x[:batch_size,:]
        mem = x[batch_size:,:]

        x =(reference- mem)**2

        return self.proj(x)



class SSLR(nn.Module):
    def __init__(self, lr=200, use_srn= True):
        super(SSLR, self).__init__()
        self.use_srn = use_srn
        element_dims=10
        set_size=16
        self.set_generator = SetGen(element_dims, set_size, lr, use_srn)
        self.f_match = F_match()
        self.f_reconstruct = F_reconstruct(element_dims)

    def forward(self, x, y, pool):
        all_images = torch.cat((x, y))
        x, losses = self.set_generator(all_images)

        batch_size = x.size(0) // 2

        match_dist, match_score = self.f_match(x[:batch_size,:,:], x[batch_size:,:,:], pool)
        generated_f, _ = self.f_reconstruct(x[:batch_size,:,:])

        if self.use_srn:
            return match_dist, losses, match_score, generated_f
        else:
            return match_dist, match_score, generated_f


def eval(net, batch_size, test_loader, pool, epoch, model_type):
    net.eval()
    all_loss = 0
    acc = 0
    import gc;
    for idx, data in enumerate(test_loader):
        images_x, images_y, s = data
        images_x, images_y, s = images_x.cuda(), images_y.cuda(), s.sum(1).cuda()/(64*64)

        if model_type == "srn":
            match_dist, inner_losses, match_score, re = net(images_x, images_y, pool)
        elif model_type == "mlp":
            match_dist, match_score, re = net(images_x, images_y, pool)
        else:
            match_dist = net(images_x, images_y).view(-1)

        loss = ((match_dist- s)**2).sum()
        all_loss += loss.item()

        acc += torch.abs((match_dist- s)/s).mean()
        acc = acc.detach().cpu()

        gc.collect()
    return all_loss/len(test_loader), acc/len(test_loader)



if __name__ == "__main__":

    args = get_args()
    print(args)

    train_loader = data.get_loader(data.IsoColorCircles(train=True, size=64000, n = 2), batch_size = args.batch_size)
    test_loader = data.get_loader(data.IsoColorCircles(train=False, size=4000, n = 2), batch_size = args.batch_size)

    if args.model_type == "srn":
        net = SSLR(float(args.inner_lr)).float().cuda()

        if args.resume is not None:
            print("resume from ", args.resume)
#             state_dict = torch.load("set_model_recon_0.1_l2.pt")
            state_dict = torch.load(args.resume)
            own_state = net.state_dict()
            for name, param in state_dict.items():
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)

    elif args.model_type == "mlp":
        net = SSLR(use_srn = False).float().cuda()

    else:
        assert args.model_type == "cnn"
        net = Net().float().cuda()

    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    writer = SummaryWriter(f"match_runs/{args.model_type}_lr={args.lr}_wd={args.weight_decay}_ilr={args.inner_lr}", purge_step=0, flush_secs = 10)

    running_loss = 0
    for epoch in range(1000+1):
        with mp.Pool(10) as pool:
            print(f"epoch {epoch}")

            net.train()
            running_loss = 0
            for idx, data in enumerate(train_loader):
                images_x, images_y, s = data
                images_x, images_y, s = images_x.cuda(), images_y.cuda(), s.sum(1).cuda()/(64*64)
                optimizer.zero_grad()

                if args.model_type == "srn":
                    match_dist, inner_losses, match_score, re = net(images_x, images_y, pool)
                elif args.model_type == "mlp":
                    match_dist, match_score, re = net(images_x, images_y, pool)
                else:
                    match_dist = net(images_x, images_y).view(-1)

                dist_loss = ((match_dist- s)**2).sum()

                use_set = (args.model_type == "srn") or (args.model_type == "mlp")
                if use_set:
                    match_loss = match_score.mean()
                    loss = dist_loss + 10*match_loss
                else:
                    loss = dist_loss

                if args.recon :
                    recon_loss = ((re - images)**2).mean()
                    loss += recon_loss

                if use_set:
                    writer.add_scalar("train/dist_loss", dist_loss.item(), global_step=epoch*len(train_loader) + idx)
                    writer.add_scalar("train/match_loss", match_loss.item(), global_step=epoch*len(train_loader) + idx)
                    if args.recon :
                        writer.add_scalar("train/recon_loss", recon_loss.item(), global_step=epoch*len(train_loader) + idx)
                writer.add_scalar("train/loss", loss.item(), global_step=epoch*len(train_loader) + idx)

                loss.backward()
                alpha = 0.05
                optimizer.step()


                if idx % (len(train_loader)//4) == 0:
                    if use_set:
                        if args.model_type == "srn":
                            print(f"inner loss {[l.item()/args.batch_size for l in inner_losses]}")
                        print("dist_loss", dist_loss.item())
                        print("match_loss",match_loss.item())
                        if args.recon :
                            print("recon_loss",recon_loss.item())
                    print("loss",loss.item())

                running_loss += loss.item()
        print(running_loss/ len(train_loader))
        if epoch % 1 ==0:
            with mp.Pool(10) as pool:
                eval_loss, acc = eval(net, args.batch_size, test_loader, pool, epoch, args.model_type)
            print(f"eval: {eval_loss}  {acc}")
            writer.add_scalar("eval/loss", eval_loss, global_step=epoch)
            writer.add_scalar("eval/acc", acc, global_step=epoch)
            writer.flush()

        print()
        #save model
        if args.save is not None:
            torch.save(net.state_dict(), args.save)
