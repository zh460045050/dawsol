import cv2

import torch
import torch.nn as nn
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch.autograd import Variable
import random
from sklearn.cluster import k_means

import importlib

import voc12.dataloader
from misc import pyutils, torchutils
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
import os

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]


def color_pro(pro, img=None, mode='hwc'):
	H, W = pro.shape
	pro_255 = (pro*255).astype(np.uint8)
	pro_255 = np.expand_dims(pro_255,axis=2)
	color = cv2.applyColorMap(pro_255,cv2.COLORMAP_JET)
	color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
	if img is not None:
		rate = 0.5
		if mode == 'hwc':
			assert img.shape[0] == H and img.shape[1] == W
			color = cv2.addWeighted(img,rate,color,1-rate,0)
		elif mode == 'chw':
			assert img.shape[1] == H and img.shape[2] == W
			img = np.transpose(img,(1,2,0))
			color = cv2.addWeighted(img,rate,color,1-rate,0)
			color = np.transpose(color,(2,0,1))
	else:
		if mode == 'chw':
			color = np.transpose(color,(2,0,1))	
	return color


def generate_vis(p, img):
    # All the input should be numpy.array 
    # img should be 0-255 uint8

    C = 1
    H, W = p.shape

    prob = p

    prob[prob<=0] = 1e-7

    def ColorCAM(prob, img):
        C = 1
        H, W = prob.shape
        colorlist = []
        colorlist.append(color_pro(prob,img=img,mode='chw'))
        CAM = np.array(colorlist)/255.0
        return CAM

    #print(prob.shape, img.shape)
    CAM = ColorCAM(prob, img)
    #print(CAM.shape)
    return CAM[0, :, :, :]

class TSA(object):

    def __init__(self, num_classes, feature_dims, sample_num_source=32, sample_num_target=32):
        self.num_classes = num_classes 
        self.feature_dims = feature_dims
        self.cluster_centers = np.zeros( (self.num_classes, self.feature_dims) ) #M_{:,1:}
        self.cluster_counts = np.zeros( (self.num_classes, 1) ) #r_{1:}
        self.universe_centers = np.zeros( (1, self.feature_dims) ) #M_{:,0}
        self.universe_count = 0 #r_{0}
        self.sample_num_source = sample_num_source
        self.sample_num_target = sample_num_target

    def samples_split(self, image_feature, pixel_feature, target):
        ##2048
        ##2048*h*w

        source = image_feature.view(image_feature.shape[0])
        mask = np.zeros( (pixel_feature.shape[1], pixel_feature.shape[2]) )
        batch_source = torch.zeros(self.sample_num_source, image_feature.shape[0]).cuda()
        batch_target = torch.zeros(self.sample_num_target, image_feature.shape[0]).cuda()
        batch_label = torch.zeros(1).cuda()
        count = 0
        count_u = 0

        batch_source[0, :] = source
        lab = target
        all_samples = pixel_feature.permute(1, 2, 0).contiguous().view(-1, source.shape[0])
        samples = all_samples.detach().clone().cpu().numpy()

        if self.cluster_counts[lab] == 0:
            self.cluster_counts[lab] = self.cluster_counts[lab] + 1
            self.cluster_centers[lab, :] = source.unsqueeze(0).detach().clone().cpu().numpy() / self.cluster_counts[lab] + self.cluster_centers[lab, :] * ((self.cluster_counts[lab]-1) / self.cluster_counts[lab] )

        #### Kmeans Cluster
        center_inits = torch.cat( [torch.from_numpy(self.universe_centers).cuda(), source.unsqueeze(0), torch.from_numpy(self.cluster_centers)[lab, :].unsqueeze(0).cuda()], dim=0).detach().clone().cpu().numpy()
        center, label, pb = k_means(samples, n_clusters=3, init=center_inits, n_init=1, random_state=0)

        #### Update Cache Matrix
        self.cluster_counts[lab] = self.cluster_counts[lab] + 1
        self.cluster_centers[lab, :] = np.expand_dims(center[1, :], axis=0) / self.cluster_counts[lab] + self.cluster_centers[lab, :] * ((self.cluster_counts[lab] - 1) / self.cluster_counts[lab] )

        self.universe_count = self.universe_count + 1
        self.universe_centers[0, :] = np.expand_dims(center[0, :], axis=0) / self.universe_count + self.universe_centers[0, :] * ((self.universe_count - 1) / self.universe_count )

        #### Sample Source/Target/Univers Items
        cur_univer = all_samples[label == 0, :]
        cur_source = all_samples[label == 1, :]
        cur_target = all_samples[label == 2, :]

        #### Random Sampling
        rand_index_target = np.arange(0, cur_target.shape[0])
        random.shuffle(rand_index_target)
        rand_index_source = np.arange(0, cur_source.shape[0])
        random.shuffle(rand_index_source)

        #print(cur_source.shape, cur_target.shape, cur_univer.shape)

        if len(rand_index_target) >= self.sample_num_target and len(rand_index_source) >= (self.sample_num_source - 1):
            batch_source[1:, :] = torch.index_select(cur_source, 0, torch.from_numpy(rand_index_source)[:min(self.sample_num_source-1, len(rand_index_source))].cuda())
            batch_target = torch.index_select(cur_target, 0, torch.from_numpy(rand_index_target)[:min(self.sample_num_target, len(rand_index_target))].cuda())
            batch_label = target
            count = count + 1
        if count_u == 0:
            batch_universum = cur_univer.view(-1, image_feature.shape[0])
            count_u = batch_universum.shape[1]
        else:
            batch_universum = torch.cat( [batch_universum, cur_univer.view(-1, image_feature.shape[0])], dim=0)

        label = np.float32(label)
        label[label == 2] = 0.5
        mask = label.reshape( (pixel_feature.shape[1], pixel_feature.shape[2]) )

        #print(count)
        return batch_source, batch_target, batch_universum, batch_label, count, mask
        
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    ####
    if source.size()[0] != target.size()[0]:
        if source.size()[0] < target.size()[0]:
            source = source.unsqueeze(0)
            source = source.expand( ( np.int64(target.size()[0] / source.size()[1]), source.size()[1], source.size()[2]))
            source = source.contiguous().view(target.size())
        else:
            target = target.unsqueeze(0)
            target = target.expand( (np.int64(source.size()[0] / target.size()[1]), target.size()[1], target.size()[2]))
            target = target.contiguous().view(source.size())
    ####
    
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


def cal_mmd(batch_source, batch_target, count):

    loss_mmd = Variable(torch.zeros(1)).cuda()
        #if self.sample_num_target == self.sample_num_source:
    loss_mmd += mmd_rbf_accelerate(batch_source, batch_target)
        #else:
        #    loss_mud += mmd_rbf_noaccelerate(batch_source[i, :, :], batch_target[i, :, :])
    #loss_mmd /= batch_source.shape[1]

    return loss_mmd

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x, _, _ = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss')))

    return


def run(args):

    model = getattr(importlib.import_module("net.resnet50_cam_dawsol"), 'Net')()

    tsa = TSA(20, 2048)


    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()
    ce = nn.CrossEntropyLoss()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            img = img.cuda()
            label = pack['label'].cuda(non_blocking=True)
            x, features, image_features = model(img)

            image_features = torchutils.gap2d(features, keepdims=True)

            loss_d = torch.zeros(1).cuda()
            loss_u = torch.zeros(1).cuda()
            #####
            if ep > 0:
                #cams = normalize_tensor(cams + 1e-10)
                labels = [torch.nonzero(label[i]).squeeze(1) for i in range(label.shape[0])]
                image_size = img.shape[2:]
                #image = img[0, :, :, :]
                count = 0
                for image, image_feature, pixel_feature, lab in zip(img, image_features, features, labels):
                    if lab.shape[0] != 1:
                        continue
                    flag_vis = count
                    S_T_f, T_t, T_u, _, flag, mask = tsa.samples_split(image_feature, pixel_feature, lab[0])
                    if flag != 0:
                        loss_d += mmd_rbf_accelerate(S_T_f, T_t) #/ image_features.shape[0]
                        '''
                        cam_normalized = cv2.resize(mask, image_size,
                                                    interpolation=cv2.INTER_CUBIC)
                        
                        if flag_vis == 0:
                            vis_image = image.cpu().data * np.array(_IMAGENET_STDDEV).reshape([3, 1, 1]) + np.array(_IMAGENET_MEAN).reshape([3, 1, 1])
                            vis_image = np.int64(vis_image * 255)
                            vis_image[vis_image > 255] = 255
                            vis_image[vis_image < 0] = 0
                            vis_image = np.uint8(vis_image)
                            plt.imsave(os.path.join(args.vis_cluster_dir, str(step) + ".png"), generate_vis(cam_normalized, vis_image).transpose(1, 2, 0))
                            #count = count + 1
                        count = count + 1
                        '''
                    loss_u += torch.mean( torch.mean(T_u)) #/ image_features.shape[0]
                if count != 0:
                    loss_d /= count
                loss_u /= features.shape[0]
                '''
                for pixel_feature, pixel_mask, lab in zip(features, cams, labels):
                    if lab.shape[0] == 0:
                        continue
                    flag_vis = count
                    for i in range(0, lab.shape[0]):
                        print(pixel_mask[lab[i]].unsqueeze(0).shape)
                        image_feature = torch.sum(pixel_mask[lab[i]].unsqueeze(0) * pixel_feature, dim=[1, 2])
                        #print(image_feature.shape)
                        S_T_f, T_t, T_u, _, flag, mask = tsa.samples_split(image_feature, pixel_feature, lab[i])
                        if flag != 0:
                            loss_d += mmd_rbf_accelerate(S_T_f, T_t) / lab.shape[0]
                            cam_normalized = cv2.resize(mask, image_size,
                                                    interpolation=cv2.INTER_CUBIC)
                            if flag_vis == 0:
                                plt.imsave(os.path.join(args.vis_cluster_dir, str(step) + "_" + str(lab[i]) + ".png"), generate_vis(cam_normalized, vis_image).transpose(1, 2, 0))
                                count = count + 1
                        loss_u += torch.mean( torch.mean(T_u)) / lab.shape[0]
                '''

            #####

            optimizer.zero_grad()
            
            loss_ce = F.multilabel_soft_margin_loss(x, label)

            #print(loss, loss_d, loss_u)
            ##
            loss = loss_ce + args.rate_d * loss_d + args.rate_u * loss_u
            ##

            loss.backward()
            avg_meter.add({'loss': loss.item()})
            avg_meter.add({'loss_ce': loss_ce.item()})
            avg_meter.add({'loss_d': loss_d.item()})
            avg_meter.add({'loss_u': loss_u.item()})


            optimizer.step()
            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss')),
                      'loss_ce:%.4f' % (avg_meter.pop('loss_ce')),
                      'loss_d:%.4f' % (avg_meter.pop('loss_d')),
                      'loss_u:%.4f' % (avg_meter.pop('loss_u')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        
        validate(model, val_data_loader)
        timer.reset_stage()

    torch.save(model.module.state_dict(), args.cam_weights_name)
    torch.cuda.empty_cache()
