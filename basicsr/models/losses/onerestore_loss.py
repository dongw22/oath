import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from math import exp
from torchvision import transforms
from torchvision.models import vgg16
import torchvision
from collections import OrderedDict
'''
MS-SSIM Loss
'''

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

## to do: the composite degradation restoration loss
def data_process(data):
    # combine_type = args.degr_type
    b,n,c,w,h = data.size()

    pos_data = data[:,0,:,:,:]

    inp_data = torch.zeros((b,c,w,h))
    inp_class = []

    neg_data = torch.zeros((b,n-2,c,w,h))

    index = np.random.randint(1, n, (b))
    for i in range(b):
        k = 0
        for j in range(n):
            if j == 0:
                continue
            elif index[i] == j:
                inp_class.append(combine_type[index[i]])
                inp_data[i, :, :, :] = data[i, index[i], :, :,:]
            else:
                neg_data[i,k,:,:,:] = data[i, j, :, :,:]
                k=k+1
    return pos_data.to("cuda" if torch.cuda.is_available() else "cpu"), inp_data.to("cuda" if torch.cuda.is_available() else "cpu"), neg_data.to("cuda" if torch.cuda.is_available() else "cpu")


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])  #算出总共求了多少次差
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个            
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.model = vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT)
        self.model = self.model.features[:16].to("cuda" if torch.cuda.is_available() else "cpu")
        for param in self.model.parameters():
            param.requires_grad = False
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def gen_features(self, x):
        output = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output
    def forward(self, inp, pos, neg, out):
        inp_t = inp
        inp_x0 = self.gen_features(inp_t)
        pos_t = pos
        pos_x0 = self.gen_features(pos_t)
        out_t = out
        out_x0 = self.gen_features(out_t)
        neg_t, neg_x0 = [],[]
        for i in range(neg.shape[1]):
            neg_i = neg[:,i,:,:]
            neg_t.append(neg_i)
            neg_x0_i = self.gen_features(neg_i)
            neg_x0.append(neg_x0_i)
        loss = 0
        for i in range(len(pos_x0)):
            pos_term = self.l1(out_x0[i], pos_x0[i].detach())
            inp_term = self.l1(out_x0[i], inp_x0[i].detach())/(len(neg_x0)+1)
            neg_term = sum(self.l1(out_x0[i], neg_x0[j][i].detach()) for j in range(len(neg_x0)))/(len(neg_x0)+1)
            loss = loss + pos_term / (inp_term+neg_term+1e-7)
        return loss / len(pos_x0)

class Total_loss(nn.Module):
    def __init__(self, args):
        super(Total_loss, self).__init__()
        self.con_loss = ContrastLoss()
        self.weight_sl1, self.weight_msssim, self.weight_drl = args.loss_weight

    def forward(self, inp, pos, neg, out):
        smooth_loss_l1 = F.smooth_l1_loss(out, pos)
        msssim_loss = 1-msssim(out, pos, normalize=True)
        c_loss = self.con_loss(inp[0], pos, neg, out)

        total_loss = self.weight_sl1 * smooth_loss_l1 + self.weight_msssim * msssim_loss + self.weight_drl * c_loss
        return total_loss