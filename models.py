import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import hungarian_loss_each
from dspn import DSPN
from fspool import FSPool


class Encoder(nn.Module):

    def __init__(self, element_dims, set_size, out_size):
        super(Encoder, self).__init__()
        self.nef = 64
        self.e_ksize = 4
        self.set_size = set_size
        self.out_size = out_size
        self.element_dims = element_dims

        self.conv1 = nn.Conv2d(3, self.nef, self.e_ksize, stride = 2, padding = 1, bias = False)

        self.conv2 = nn.Conv2d(self.nef, self.nef*2, self.e_ksize, stride = 2, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(self.nef*2)

        self.conv3 = nn.Conv2d(self.nef*2, self.nef*4, self.e_ksize, stride = 2, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.nef*4)

        self.conv4 = nn.Conv2d(self.nef*4, self.nef*8, self.e_ksize, stride = 4, padding = 1, bias = False)


        self.bn4 = nn.BatchNorm2d(self.nef*8)

        self.proj = nn.Linear(self.nef*32, self.out_size)

        self.proj_s = nn.Conv1d(2048//self.set_size, self.element_dims, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))

        s = self.proj_s(out.view(out.shape[0],  self.set_size, 2048//self.set_size).transpose(1,2))

        out = out.view(out.shape[0], self.nef*32)
        return self.proj(out), s


class Decoder(nn.Module):

    def __init__(self, input_dim):
        super(Decoder, self).__init__()

        self.ngf = 256
        g_ksize = 4
        self.proj = nn.Linear(input_dim, self.ngf * 4 * 4 * 4)
        self.bn0 = nn.BatchNorm1d(self.ngf * 4 * 4 * 4)

        self.dconv1 = nn.ConvTranspose2d(self.ngf * 4,self.ngf*2, g_ksize,
            stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ngf*2)

        self.dconv2 = nn.ConvTranspose2d(self.ngf*2, self.ngf, g_ksize,
            stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.ngf)

        self.dconv3 = nn.ConvTranspose2d(self.ngf, 3, g_ksize,
            stride=4, padding=0, bias=False)

    def forward(self, z, c=None):
        out = F.relu(self.bn0(self.proj(z)).view(-1, self.ngf* 4, 4, 4))
        out = F.relu(self.bn1(self.dconv1(out)))
        out = F.relu(self.bn2(self.dconv2(out)))
        out =  self.dconv3(out)
        return out


class FSEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, output_channels, 1),
        )
        self.pool = FSPool(output_channels, 20, relaxed=False)

    def forward(self, x, mask=None):
        x = self.conv(x)
        x = x / x.size(2)  # normalise so that activations aren't too high with big sets
        x, _ = self.pool(x)
        return x


class SetGen(nn.Module):
    def __init__(self, element_dims=10, set_size=16, lr=200, use_srn= True, iters=5):
        super(SetGen, self).__init__()
        self.use_srn = use_srn
        CNN_ENCODER_SPACE = 100
        # H_{agg}
        self.encoder = FSEncoder(element_dims, CNN_ENCODER_SPACE, 512)
        self.decoder = DSPN(
                    self.encoder, element_dims, iters=iters, lr=lr
                )
        # H_{set} and H_{embed}
        self.img_encoder = Encoder(element_dims, set_size, CNN_ENCODER_SPACE)

    def forward(self, x):
        x, s = self.img_encoder(x)
        if self.use_srn:
            intermediate_sets, losses, grad_norms = self.decoder(x, s)
            x = intermediate_sets[-1]
        else:
            x = s

        if self.use_srn:
            return x, losses
        else:
            return x, None


class F_match(nn.Module):
    def __init__(self):
        super(F_match, self).__init__()
        self.proj1 = torch.nn.Conv1d(10, 3, 1)
        self.proj2 = torch.nn.Conv1d(10, 3, 1)

    def forward(self, x_set, y_set, pool):
        # x_set shape: B, element_dims, set_size
        x_att = self.proj1(x_set)
        y_att = self.proj1(y_set)

        x_loc = self.proj2(x_set)
        y_loc = self.proj2(y_set)

        # matching
        indices = hungarian_loss_each(x_att, y_att, pool)
        l = [
            (x_loc[idx,:,row_idx] - y_loc[idx,:,col_idx])**2
            for idx, (row_idx, col_idx) in enumerate(indices)
        ]
        l_m = [
            ((x_att[idx,:,row_idx] - y_att[idx,:,col_idx])**2).sum()
            for idx, (row_idx, col_idx) in enumerate(indices)
        ]
        match_dist = torch.stack(list(l)).sum(1).sum(1)
        match_score = torch.stack(list(l_m))
        return match_dist, match_score


class F_reconstruct(nn.Module):
    def __init__(self, element_dims=10):
        super(F_reconstruct, self).__init__()
        self.vec_decoder = Decoder(element_dims)

    def forward(self, x_set):
        batch_size = x_set.size(0)
        element_dims = x_set.size(1)
        set_size = x_set.size(2)

        x = x_set.transpose(1,2).reshape(-1,element_dims)
        generated = self.vec_decoder(x)
        generated = generated.reshape(batch_size, set_size, 3, 64, 64)

        attention = torch.softmax(generated, dim=1)
        generated_set = torch.sigmoid(generated)

        generated_set = generated_set*attention
        generated_f = generated_set.sum(dim=1).clamp(0,1)

        return generated_f, generated_set


class EncoderCLEVR(nn.Module):

    def __init__(self, element_dims=10, set_size=16, out_size=512):
        super(EncoderCLEVR, self).__init__()
        self.nef = 64
        self.e_ksize = 4
        self.set_size = set_size
        self.out_size = out_size
        self.element_dims = element_dims

        self.conv1 = nn.Conv2d(3, self.nef, self.e_ksize, stride = 2, padding = 1, bias = False)

        self.conv2 = nn.Conv2d(self.nef, self.nef*2, self.e_ksize, stride = 2, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(self.nef*2)

        self.conv3 = nn.Conv2d(self.nef*2, self.nef*4, self.e_ksize, stride = 2, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.nef*4)

        self.conv4 = nn.Conv2d(self.nef*4, self.nef*8, self.e_ksize, stride = 4, padding = 1, bias = False)
        self.bn4 = nn.BatchNorm2d(self.nef*8)

        self.proj = nn.Linear(self.nef*128, self.out_size)
        self.proj_s = nn.Conv1d(8192//self.set_size, self.element_dims, 1)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))

        s = self.proj_s(out.view(out.shape[0], self.set_size, 8192//self.set_size).transpose(1,2))

        out = out.view(out.shape[0], self.nef*128)
        return self.proj(out), s


class DecoderCLEVR(nn.Module):
    def __init__(self, input_dim):
        super(DecoderCLEVR, self).__init__()

        self.ngf = 256
        g_ksize = 4
        self.proj = nn.Linear(input_dim, self.ngf * 4 * 4 * 4 * 4)
        self.bn0 = nn.BatchNorm1d(self.ngf * 4 * 4 * 4 * 4)

        self.dconv1 = nn.ConvTranspose2d(self.ngf * 4,self.ngf*2, g_ksize,
            stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ngf*2)

        self.dconv2 = nn.ConvTranspose2d(self.ngf*2, self.ngf, g_ksize,
            stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.ngf)

        self.dconv3 = nn.ConvTranspose2d(self.ngf, 3, g_ksize,
            stride=4, padding=0, bias=False)

    def forward(self, z, c=None):
        out = F.relu(self.bn0(self.proj(z)).view(-1, self.ngf* 4, 4*2, 4*2))
        out = F.relu(self.bn1(self.dconv1(out)))
        out = F.relu(self.bn2(self.dconv2(out)))
        out =  self.dconv3(out)
        return out


class F_reconstruct_CLEVR(nn.Module):
    def __init__(self, element_dims=10):
        super(F_reconstruct_CLEVR, self).__init__()
        self.vec_decoder = DecoderCLEVR(element_dims)

    def forward(self, x_set):
        batch_size = x_set.size(0)
        element_dims = x_set.size(1)
        set_size = x_set.size(2)

        x = x_set.transpose(1,2).reshape(-1,element_dims)
        generated = self.vec_decoder(x)
        generated = generated.reshape(batch_size, set_size, 3, 128, 128)

        attention = torch.softmax(generated, dim=1)
        generated_set = torch.sigmoid(generated)

        generated_set = generated_set*attention
        generated_f = generated_set.sum(dim=1).clamp(0,1)

        return generated_f, generated_set


class RNFSEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * input_channels, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, output_channels, 1),
        )
        self.lin = nn.Linear(dim, output_channels)
        self.pool = FSPool(output_channels, 20, relaxed=False)

    def forward(self, x, mask=None):
        # create all pairs of elements
        x = torch.cat(utils.outer(x), dim=1)
        x = self.conv(x)
        # flatten pairs and scale appropriately
        n, c, l, _ = x.size()
        x = x.view(x.size(0), x.size(1), -1) / l / l
        x, _ = self.pool(x)
        return x


class SetGenCLEVR(nn.Module):
    def __init__(self, element_dims=10, set_size=16, lr=8, use_srn=True):
        super(SetGenCLEVR, self).__init__()
        self.use_srn = use_srn
        CNN_ENCODER_SPACE = 512
        # H_{agg}
        self.encoder = RNFSEncoder(element_dims, CNN_ENCODER_SPACE, 512)
        self.decoder = DSPN(
                    self.encoder, element_dims, iters=10, lr=lr
                )
        # H_{set} and H_{embed}
        self.img_encoder = EncoderCLEVR(element_dims, set_size, CNN_ENCODER_SPACE)

    def forward(self, x):
        x, s = self.img_encoder(x)
        if self.use_srn:
            intermediate_sets, losses, grad_norms = self.decoder(x, s)
            x = intermediate_sets[-1]
        else:
            x = s

        if self.use_srn:
            return x, losses
        else:
            return x, None

