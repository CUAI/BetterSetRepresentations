import scipy
import scipy.optimize
import torch
import torch.nn.functional as F
import numpy as np
import cv2

def hungarian_loss_each(predictions, targets, thread_pool):
    # predictions and targets shape :: (n, c, s)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (n, s, s)
    squared_error = F.smooth_l1_loss(predictions, targets, reduction="none").mean(1)

    squared_error_np = squared_error.detach().cpu().numpy()

    indices = thread_pool.map(hungarian_loss_per_sample, squared_error_np)
    return indices

def hungarian_loss_per_sample(sample_np):
    return scipy.optimize.linear_sum_assignment(sample_np)


def chamfer_loss(predictions, targets):
    # predictions and targets shape :: (k, n, c, s)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (k, n, s, s)
    squared_error = F.smooth_l1_loss(predictions, targets, reduction="none").mean(2)
    loss = squared_error.min(2)[0] + squared_error.min(3)[0]
    return loss.view(loss.size(0), -1).mean(1)


def chamfer_loss_each(predictions, targets):
    # predictions and targets shape :: (k, n, c, s)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (k, n, s, s)
    squared_error = F.smooth_l1_loss(predictions, targets, reduction="none").mean(2)
    return torch.cat((squared_error.min(2)[0], squared_error.min(3)[0]),2)[0]



def scatter_masked(tensor, mask, p, binned=False, threshold=None):
    s = tensor[0].detach().cpu()
    mask = mask[0].detach().clamp(min=0, max=1).cpu()
    p = p[0].detach().clamp(min=0, max=1).cpu()
    if binned:
        s = s * 128
        s = s.view(-1, s.size(-1))
        mask = mask.view(-1)
    if threshold is not None:
        keep = mask.view(-1) > threshold
        s = s[:, keep]
        mask = mask[keep]
    return s, mask, p


def cv_bbox(np_imgs):
    imgs = []
    for np_img in np_imgs:
        new_img = np_img.copy()
        cnts = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            new_img[y:y+h, x:x+w] = 1
        imgs.append(new_img)
    return torch.tensor(imgs).reshape(-1,16,128,128)


def chamfer_score(s1, s2, SMOOTH=1e-6):
    batch = s1.size(0)
    size = s1.size(1)
    a = torch.cat(size*[s1.unsqueeze(1)],1).reshape(-1, 128,128)
    b = torch.cat(size*[s2.unsqueeze(2)],2).reshape(-1, 128,128)
    
    intersect = (a & b).sum((1,2)).float()
    union = (a | b).sum((1,2)).float()
    iou = ((intersect + SMOOTH) / (union + SMOOTH))
    
    r = iou.reshape(batch, size,size, -1).squeeze(3)

    return r.max(2)[0].mean()


def outer(a, b=None):
    """ Compute outer product between a and b (or a and a if b is not specified). """
    if b is None:
        b = a
    size_a = tuple(a.size()) + (b.size()[-1],)
    size_b = tuple(b.size()) + (a.size()[-1],)
    a = a.unsqueeze(dim=-1).expand(*size_a)
    b = b.unsqueeze(dim=-2).expand(*size_b)
    return a, b
