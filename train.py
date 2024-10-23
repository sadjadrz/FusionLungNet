import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
import os
import numpy as np
import glob

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import FusionLungNet

import pytorch_ssim
import pytorch_iou
import pytoch_focal

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
focal_loss = pytoch_focal.BinaryFocalLoss()

# def focal_loss(yhat, ytrue, alpha=0.75, gamma=2):
#     # compute the actual focal loss
#     focal = -alpha * torch.pow(1. - yhat, gamma) * torch.log(yhat)
#     f_loss = torch.sum(ytrue * focal, dim=1)
#
#     return torch.mean(f_loss)

def bce_ssim_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    focal_out = focal_loss(pred, target)
    loss = ssim_out + iou_out + focal_out

    return loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):
    loss0 = bce_ssim_loss(d0, labels_v)
    loss1 = bce_ssim_loss(d1, labels_v)
    loss2 = bce_ssim_loss(d2, labels_v)
    loss3 = bce_ssim_loss(d3, labels_v)
    loss4 = bce_ssim_loss(d4, labels_v)
    loss5 = bce_ssim_loss(d5, labels_v)
    loss6 = bce_ssim_loss(d6, labels_v)
    loss7 = bce_ssim_loss(d7, labels_v)
    # ssim0 = 1 - ssim_loss(d0,labels_v)

    # iou0 = iou_loss(d0,labels_v)
    # loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7  # + 5.0*lossa
    #print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    #loss0.data[0], loss1.data[0], loss2.data[0], loss3.data[0], loss4.data[0], loss5.data[0], loss6.data[0]))
    # print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[
    # 0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    loss0.data, loss1.data, loss2.data, loss3.data, loss4.data, loss5.data, loss6.data))
    return loss0, loss

def muti_bce_loss_fusion1( d2, d3, d4, d5, labels_v):
    loss2 = bce_ssim_loss(d2, labels_v)
    loss3 = bce_ssim_loss(d3, labels_v)
    loss4 = bce_ssim_loss(d4, labels_v)
    loss5 = bce_ssim_loss(d5, labels_v)
    # ssim0 = 1 - ssim_loss(d0,labels_v)

    # iou0 = iou_loss(d0,labels_v)
    # loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
    loss = loss2 + loss3 + loss4 + loss5  # + 5.0*lossa
    #print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    #loss0.data[0], loss1.data[0], loss2.data[0], loss3.data[0], loss4.data[0], loss5.data[0], loss6.data[0]))
    # print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[
    # 0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))
    print("l2: %3f, l3: %3f, l4: %3f, l5: %3f\n" % (
     loss2.data, loss3.data, loss4.data, loss5.data))
    return loss2, loss
if __name__ == '__main__':
    # ------- 2. set the directory of training dataset --------

    data_dir = 'train_data'
    
    # Constructing paths using os.path.join
    tra_image_dir = os.path.join('content', 'lung dataset', 'train_images')
    tra_label_dir = os.path.join('content', 'lung dataset', 'train_masks')

    image_ext = '.tif'
    # label_ext = '.png'
    label_ext = '.tif'

    model_dir = os.path.join(".", "saved_models", "lung", "new_lung_copy")
    #model_dir = "./saved_models/face_mask"

    epoch_num = 40
    batch_size_train = 8
    batch_size_val = 1
    train_num = 0
    val_num = 0

    tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, '*' + image_ext))
    tra_lbl_name_list = []
    for img_path in tra_img_name_list:

        #img_name = img_path.split("/")[-1]
        img_name = img_path.split("\\")[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(os.path.join(data_dir, tra_label_dir, imidx + label_ext))

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            #RandomCrop(224),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

    # ------- 3. define model --------
    net = FusionLungNet(False)
    if torch.cuda.is_available():
        net.cuda()
        print('cuda')

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if  torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
                # inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d2, d3, d4, d5 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion1(d2, d3, d4, d5, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data
#            running_loss += loss.data[0]
            running_tar_loss += loss2.data
#            running_tar_loss += loss2.data[0]

            # del temporary outputs and loss
            del  d2, d3, d4, d5, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d]_focal4_ train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val))

            if ite_num % 250 == 0 and ite_num>300  :  # save model every 2000 iterations

                torch.save(net.state_dict(), model_dir + "fusionLungNet_bsi_focal3.3_4_itr_%d_train_%3f_tar_%3f.pth" % (
                ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

    print('-------------Congratulations! Training Done!!!-------------')
