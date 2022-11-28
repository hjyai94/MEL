import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.models import ModelBuilder
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from data_loader.dataset import TrainDataset, ValidDataset, ValDataset, norm
import os
import nibabel as nib
import argparse
from utils.util import AverageMeter
from distutils.version import LooseVersion
import math
from tensorboardX import SummaryWriter
import json
import setproctitle  # pip install setproctitle
from model.losses import bratsDiceLossOriginal5, bratsConsensusDiceLoss, bratsPredictionOverlap
from test import segment, test

def train(train_loader, net1, net2, criterion, optimizer1, optimizer2, epoch, args):
    losses1 = AverageMeter()
    losses2 = AverageMeter()

    net1.train()
    net2.train()
    for iteration, sample in enumerate(train_loader):
        image = sample['images'].float()
        target = sample['labels'].long()
        image = Variable(image).cuda()
        label = Variable(target).cuda()

        # extract the center part of the labels
        start_index = []
        end_index = []
        for i in range(3):
            start = int((args.crop_size[i] - args.center_size[i]) / 2)
            start_index.append(start)
            end_index.append(start + args.center_size[i])
        label = label[:, start_index[0]:end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]]

        # mutual learning algorithm
        ## compute out1, out2
        out1 = net1(image)
        out2 = net2(image)

        # get the same size output of two nets
        if args.net1 == 'Unet':
            crop_center_index = int((args.crop_size[0]-args.center_size[0])//2)
            out1 = out1[:, :, crop_center_index:-crop_center_index, crop_center_index:-crop_center_index, crop_center_index:-crop_center_index]

        if args.net2 == 'Unet':
            crop_center_index = int((args.crop_size[0]-args.center_size[0])//2)
            out2 = out2[:, :, crop_center_index:-crop_center_index, crop_center_index:-crop_center_index, crop_center_index:-crop_center_index]
        # extract the center part of the labels
        out1_ce = out1.permute(0, 2, 3, 4, 1).contiguous().view(-1, args.num_classes)
        out2_ce = out2.permute(0, 2, 3, 4, 1).contiguous().view(-1, args.num_classes)
        label_ce = label.contiguous().view(-1).cuda()

        out1 = F.softmax(out1, 1)
        out2 = F.softmax(out2, 1)

        # update net1
        if args.mutual_loss == "consensus_dice":
            loss1 = criterion(out1_ce, label_ce) + bratsConsensusDiceLoss(out2, out1, label)
        elif args.mutual_loss == "consensus_dice_only":
            loss1 = bratsConsensusDiceLoss(out2, out1, label)
        else:
            raise('Mutual loss is not defined')
        losses1.update(loss1.item(), image.size(0))  

        # compute gradient and do SGD step
        optimizer1.zero_grad()
        loss1.backward(retain_graph=True)
        optimizer1.step()

        # compute out1
        out1 = net1(image)

        if args.net1 == 'Unet':
            crop_center_index = int((args.crop_size[0]-args.center_size[0])//2)
            out1 = out1[:, :, crop_center_index:-crop_center_index, crop_center_index:-crop_center_index, crop_center_index:-crop_center_index]

        out1 = F.softmax(out1, 1)
        # update net2
        if args.mutual_loss == "consensus_dice":
            loss2 = criterion(out2_ce, label_ce) + bratsConsensusDiceLoss(out1, out2, label)
        elif args.mutual_loss == "consensus_dice_only":
            loss2 = bratsConsensusDiceLoss(out1, out2, label)
        elif args.mutual_loss == "kl_divergence":
            loss2 = criterion(out2_ce, label_ce) + F.kl_div(out1, out2)
        elif args.mutual_loss == "prediction_overlap":
            loss2 = criterion(out2_ce, label_ce) + bratsPredictionOverlap(out1, out2)
        elif args.mutual_loss == "no_mutual":
            loss2 = criterion(out2_ce, label_ce)
        else:
            raise ('Mutual loss is not defined')

        # losses.update(loss.data[0],image.size(0))
        losses2.update(loss2.item(), image.size(0))  # changed by hjy

        # compute gradient and do SGD step
        optimizer2.zero_grad()
        loss2.backward()
        # loss2.backward()
        optimizer2.step()

        # adjust learning rate
        cur_iter = iteration + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizer1, cur_iter, args)
        # adjust_learning_rate(optimizer2, cur_iter, args)

        print('   * i {} |  lr: {:.6f} | Training Loss1: {losses1.avg:.3f} | Training Loss2: {losses2.avg:.3f}'.format(iteration, args.running_lr,
                                                                                 losses1=losses1, losses2=losses2))

    print('   * EPOCH {epoch} | Training Loss1: {losses1.avg:.3f} | Training Loss2: {losses2.avg:.3f}'.format(epoch=epoch, losses1=losses1, losses2=losses2))

    return losses1.avg, losses2.avg

# validation in the validation dataset to compute each score for each epoch
def valid_score(net1, net2, epoch, args):
    net1.eval()
    net2.eval()
    # initialization
    num_ignore = 0
    margin = [args.crop_size[k] - args.center_size[k] for k in range(3)]
    num_images = int(len(valid_dir))
    dice_score_net1, dice_score_net2, dice_score_avg = np.zeros([num_images, 3]).astype(float), np.zeros(
        [num_images, 3]).astype(float), np.zeros([num_images, 3]).astype(float)

    for i in range(num_images):
        # load the images, label and mask
        direct, _ = valid_dir[i].split("\n")
        _, patient_ID = direct.split('/')

        if args.correction:
            flair = nib.load(os.path.join(args.root_path, direct, patient_ID + '_flair_corrected.nii.gz')).get_data()
            t2 = nib.load(os.path.join(args.root_path, direct, patient_ID + '_t2_corrected.nii.gz')).get_data()
            t1 = nib.load(os.path.join(args.root_path, direct, patient_ID + '_t1_corrected.nii.gz')).get_data()
            t1ce = nib.load(os.path.join(args.root_path, direct, patient_ID + '_t1ce_corrected.nii.gz')).get_data()
            # print('Using bias correction dataset')
        else:
            flair = nib.load(os.path.join(args.root_path, direct, patient_ID + '_flair.nii.gz')).get_data()

            t2 = nib.load(os.path.join(args.root_path, direct, patient_ID + '_t2.nii.gz')).get_data()

            t1 = nib.load(os.path.join(args.root_path, direct, patient_ID + '_t1.nii.gz')).get_data()

            t1ce = nib.load(os.path.join(args.root_path, direct, patient_ID + '_t1ce.nii.gz')).get_data()
            # print("not using bias correction correction dataset")

        mask = nib.load(os.path.join(args.root_path, direct, patient_ID + '_mask.nii.gz')).get_data()
        labels = nib.load(os.path.join(args.root_path, direct, patient_ID + '_seg.nii.gz')).get_data()
        mask = mask.astype(int)
        labels = labels.astype(int)
        flair = np.expand_dims(norm(flair), axis=0).astype(float)
        t2 = np.expand_dims(norm(t2), axis=0).astype(float)
        t1 = np.expand_dims(norm(t1), axis=0).astype(float)
        t1ce = np.expand_dims(norm(t1ce), axis=0).astype(float)
        images = np.concatenate([flair, t2, t1, t1ce], axis=0).astype(float)

        # divide the input images input small image segments
        # return the padding input images which can be divided exactly
        image_pad, mask_pad, label_pad, num_segments, padding_index, index = segment(images, mask, labels, args)

        # initialize prediction for the whole image as background
        labels_shape = list(labels.shape)
        labels_shape.append(args.num_classes)
        pred_net1, pred_net2, pred_avg = np.zeros(labels_shape), np.zeros(labels_shape), np.zeros(labels_shape)
        pred_net1[:, :, :, 0], pred_net2[:, :, :, 0], pred_avg[:, :, :, 0] = 1, 1, 1

        # initialize the prediction for a small segmentation as background
        pad_shape = [int(num_segments[k] * args.center_size[k]) for k in range(3)]
        pad_shape.append(args.num_classes)
        pred_net1_pad, pred_net2_pad, pred_avg_pad = np.zeros(pad_shape), np.zeros(pad_shape), np.zeros(pad_shape)
        pred_net1_pad[:, :, :, 0], pred_net2_pad[:, :, :, 0], pred_avg_pad[:, :, :, 0] = 1, 1, 1

        # score_per_image stores the sum of each image
        score_net1_per_image, score_net2_per_image, score_avg_per_image = np.zeros([3, 3]), np.zeros([3, 3]), np.zeros(
            [3, 3])
        # iterate over the z dimension
        for idz in range(num_segments[2]):
            tf = ValDataset(image_pad, label_pad, mask_pad, num_segments, idz, args)
            valid_loader = DataLoader(tf, batch_size=1, shuffle=args.shuffle, num_workers=args.num_workers,
                                     pin_memory=True)
            score_net1_seg, score_net2_seg, score_avg_seg, pred_seg_net1, pred_seg_net2, pred_seg_avg = \
                test(valid_loader, net1, net2, num_segments, args)
            pred_net1_pad[:, :, idz * args.center_size[2]:(idz + 1) * args.center_size[2], :] = pred_seg_net1
            pred_net2_pad[:, :, idz * args.center_size[2]:(idz + 1) * args.center_size[2], :] = pred_seg_net2
            pred_avg_pad[:, :, idz * args.center_size[2]:(idz + 1) * args.center_size[2], :] = pred_seg_avg

            score_net1_per_image += score_net1_seg
            score_net2_per_image += score_net2_seg
            score_avg_per_image += score_avg_seg

            # decide the start and end point in the original image
        for k in range(3):
            if index[0][k] == 0:
                index[0][k] = int(margin[k] / 2 - padding_index[0][k])
            else:
                index[0][k] = int(margin[k] / 2 + index[0][k])

            index[1][k] = int(min(index[0][k] + num_segments[k] * args.center_size[k], labels.shape[k]))

        dist = [index[1][k] - index[0][k] for k in range(3)]
        pred_net1[index[0][0]:index[1][0], index[0][1]:index[1][1], index[0][2]:index[1][2]] = pred_net1_pad[:dist[0],
                                                                                               :dist[1],
                                                                                               :dist[2]]
        pred_net2[index[0][0]:index[1][0], index[0][1]:index[1][1], index[0][2]:index[1][2]] = pred_net2_pad[:dist[0],
                                                                                               :dist[1],
                                                                                               :dist[2]]
        pred_avg[index[0][0]:index[1][0], index[0][1]:index[1][1], index[0][2]:index[1][2]] = pred_avg_pad[:dist[0],
                                                                                              :dist[1],
                                                                                              :dist[2]]

        if np.sum(score_net1_per_image[0, :]) == 0 or np.sum(score_net1_per_image[1, :]) == 0 or np.sum(
                score_net1_per_image[2, :]) == 0 or np.sum(score_net2_per_image[0, :]) == 0 or np.sum(
            score_net2_per_image[1, :]) == 0 or np.sum(score_net2_per_image[2, :]) == 0:
            num_ignore += 1
            continue

        # compute the Enhance, Core and Whole dice score
        net1_dice_score_per = [
            2 * np.sum(score_net1_per_image[k, 2]) / (
                        np.sum(score_net1_per_image[k, 0]) + np.sum(score_net1_per_image[k, 1])) for k in
            range(3)]
        net2_dice_score_per = [
            2 * np.sum(score_net2_per_image[k, 2]) / (
                        np.sum(score_net2_per_image[k, 0]) + np.sum(score_net2_per_image[k, 1])) for k in
            range(3)]
        avg_dice_score_per = [
            2 * np.sum(score_avg_per_image[k, 2]) / (
                        np.sum(score_avg_per_image[k, 0]) + np.sum(score_avg_per_image[k, 1])) for k in
            range(3)]

        print('Image: %d,|| net1, Enhance: %.4f, Core: %.4f, Whole: %.4f || '
              'net2, Enhance: %.4f, Core: %.4f, Whole: %.4f || '
              'ensemble, Enhance: %.4f, Core: %.4f, Whole: %.4f' % (
                  i, net1_dice_score_per[0], net1_dice_score_per[1], net1_dice_score_per[2],
                  net2_dice_score_per[0], net2_dice_score_per[1], net2_dice_score_per[2],
                  avg_dice_score_per[0], net2_dice_score_per[1], net2_dice_score_per[2]))

        dice_score_net1[i - num_ignore, :] = net1_dice_score_per
        dice_score_net2[i - num_ignore, :] = net2_dice_score_per
        dice_score_avg[i - num_ignore, :] = avg_dice_score_per

    count_image = num_images - num_ignore
    dice_score_net1 = dice_score_net1[:count_image, :]
    mean_net1_dice = np.mean(dice_score_net1, axis=0)
    std_net1_dice = np.std(dice_score_net1, axis=0)

    dice_score_net2 = dice_score_net2[:count_image, :]
    mean_net2_dice = np.mean(dice_score_net2, axis=0)
    std_net2_dice = np.std(dice_score_net2, axis=0)

    dice_score_avg = dice_score_avg[:count_image, :]
    mean_ensemble_dice = np.mean(dice_score_avg, axis=0)
    std_ensemble_dice = np.std(dice_score_avg, axis=0)

    return mean_net1_dice, mean_net2_dice, mean_ensemble_dice

def save_checkpoint(state, epoch, args, net_name):
    filename = args.ckpt + '/'+ net_name+ '_' + str(epoch) + '_checkpoint.pth.tar'
    print(filename)
    torch.save(state, filename)


def adjust_learning_rate(optimizer, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    # import network architecture
    builder = ModelBuilder()
    net1 = builder.build_net(
        arch=args.net1,
        num_input=args.num_input,
        num_classes=args.num_classes)
    net2 = builder.build_net(
        arch=args.net2,
        num_input=args.num_input,
        num_classes=args.num_classes)
    net1 = torch.nn.DataParallel(net1).cuda()
    net2 = torch.nn.DataParallel(net2).cuda()
    cudnn.benchmark = True

    # collect the number of parameters in the network
    print("------------------------------------------")
    print("Network Architecture of Model %s:" % (args.net1))
    num_para = 0
    for name, param in net1.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul

    print(net1)
    print("Number of trainable parameters %d in Model %s" % (num_para, args.net1))
    print("------------------------------------------")

    # set the optimizer and loss
    optimizer1 = optim.RMSprop(net1.parameters(), args.lr, alpha=args.alpha, eps=args.eps,
                              weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer2 = optim.RMSprop(net2.parameters(), args.lr, alpha=args.alpha, eps=args.eps,
                              weight_decay=args.weight_decay, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    if args.resume_net1 and args.resume_net2:
        if os.path.isfile(args.resume_net1) and os.path.isfile(args.resume_net2):
            print("=> Loading checkpoint '{}'".format(args.resume_net1))
            print("=> Loading checkpoint '{}'".format(args.resume_net2))

            checkpoint_net1 = torch.load(args.resume_net1)
            state_dict_net1 = checkpoint_net1['state_dict']
            net1.load_state_dict(state_dict_net1)

            checkpoint_net2 = torch.load(args.resume_net2)
            state_dict_net2 = checkpoint_net2['state_dict']
            net2.load_state_dict(state_dict_net2)
            print("=> Loaded checkpoint (epoch {})".format(checkpoint_net1['epoch']))
        else:
            raise Exception("=> No checkpoint found at '{}'".format(args.resume_net1))

    # loading data
    tf = TrainDataset(train_dir, args)
   
    train_loader = DataLoader(tf, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers,
                              pin_memory=True)
    if args.tensorboard:
        writer = SummaryWriter()

    print("Start training ...")
    # for epoch in range(args.start_epoch + 1, args.num_epochs + 1):
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        setproctitle.setproctitle("Epoch:{}/{}".format(epoch,args.num_epochs))
        loss1_avg, loss2_avg = train(train_loader, net1, net2, criterion, optimizer1, optimizer2, epoch, args)
        if args.tensorboard:
            writer.add_scalar('./saved/log/loss1', loss1_avg, epoch)
            writer.add_scalar('./saved/log/loss2', loss2_avg, epoch)
        if epoch % 50 == 0 or (epoch >= 400 and epoch % 25 == 0):  # validation every 50 epoch
            with torch.no_grad():
                mean_dice_net1, mean_dice_net2, mean_dice_ensem = valid_score(net1, net2, epoch, args)
                torch.cuda.empty_cache()

                if args.tensorboard:
                    writer.add_scalars('net1 scores', {'Enhance Score': mean_dice_net1[0],
                                                  'Core Score': mean_dice_net1[1], 'Whole Score': mean_dice_net1[2]}, epoch)
                    writer.add_scalars('net2 scores', {'Enhance Score': mean_dice_net2[0],
                                                       'Core Score': mean_dice_net2[1],
                                                       'Whole Score': mean_dice_net2[2]}, epoch)
                    writer.add_scalars('ensem scores', {'Enhance Score': mean_dice_ensem[0],
                                                       'Core Score': mean_dice_ensem[1],
                                                       'Whole Score': mean_dice_ensem[2]}, epoch)
        # save models
        if epoch > args.particular_epoch:
            if epoch % args.save_epochs_steps == 0:
                save_checkpoint({'epoch': epoch, 'state_dict': net1.state_dict(), 'opt_dict': optimizer1.state_dict()},
                                epoch, args, net_name='net1')
                save_checkpoint({'epoch': epoch, 'state_dict': net2.state_dict(), 'opt_dict': optimizer2.state_dict()},
                                epoch, args, net_name='net2')
    print("Training Done")

    # export scalar data to JSON for external processing
    # writer.export_scalars_to_json("./saved/log/all_scalars.json")
    # writer.close()

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.3.0'), \
        'PyTorch>=0.3.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='MEL',
                        help='a name for identitying the architecture.')
    parser.add_argument('--net1', default='Unet',
                        help='a name for identitying the subnet. Choose from the following options: DM, Unet.')
    parser.add_argument('--net2', default='Basic',
                        help='a name for identitying the subnet. Choose from the following options: DM, Unet.')
    parser.add_argument('--loss', default='cross_entropy',
                        help='a name of loss function. ')
    parser.add_argument('--mutual_loss', default='consensus_dice',
                        help='a name of loss function. choose from concensus_dice or kl_divergence, prediction_overlap,consensus_dice_only')

    # Path related arguments
    parser.add_argument('--train_path', default='datalist/train.txt',
                        help='text file of the name of training data')
    parser.add_argument('--valid_path', default='datalist/test.txt',
                        help='text file of the name of validation data')
    parser.add_argument('--root_path', default='/hjy/Dataset/MICCAI_BraTS_2018_Data_Training',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved/models',
                        help='folder to output checkpoints')

    # Data related arguments
    parser.add_argument('--crop_size', default=[64, 64, 64], nargs='+', type=int,
                        help='crop size of the input image (int or list)')
    parser.add_argument('--center_size', default=[36, 36, 36], nargs='+', type=int,
                        help='the corresponding output size of the input image (int or list)')
    parser.add_argument('--num_classes', default=5, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=5, type=int,
                        help='number of input image for each patient include four modalities and the mask')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--random_flip', default=True, type=bool,
                        help='if horizontally flip images when training')
    parser.add_argument('--random_augment', default=False, type=bool,
                        help='randomly augment data')
    parser.add_argument('--normalization', default=True, type=bool,
                        help='normalizae the data before training')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='if shuffle the data during training')
    parser.add_argument('--mask', default=True, type=bool,
                        help='if have the mask')

    # optimization related arguments
    parser.add_argument('--gpu', default='1', type=str, help='Supprot one GPU & multiple GPUs.')
    parser.add_argument('--batch_size', default=6, type=int,
                        help='training batch size')
    parser.add_argument('--num_epochs', default=1000, type=int,
                        help='epochs for training')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='start learning rate')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop learning rate')
    parser.add_argument('--optim', default='RMSprop', help='optimizer')
    parser.add_argument('--alpha', default='0.9', type=float, help='alpha in RMSprop')
    parser.add_argument('--eps', default=10 ** (-4), type=float, help='eps in RMSprop')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weights regularizer')
    parser.add_argument('--momentum', default=0.6, type=float, help='momentum for RMSprop')
    parser.add_argument('--save_epochs_steps', default=10, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--particular_epoch', default=200, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')
    parser.add_argument('--resume_net1', default='',
                        help='the checkpoint that resumes from')
    parser.add_argument('--resume_net2', default='',
                        help='the checkpoint that resumes from')
    parser.add_argument('--num_round', default=1, type=int)
    parser.add_argument('--correction', dest='correction', type=bool, default=False)
    parser.add_argument('--tensorboard', action='store_true', help='save tensorboard file')

    args = parser.parse_args()


    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    #open training file
    train_file = open(args.train_path, 'r')
    train_dir = train_file.readlines()
    # open validation file
    valid_file = open(args.valid_path, 'r')
    valid_dir = valid_file.readlines()

    print('numbers of patient Id', len(train_dir))
    args.ckpt = os.path.join(args.ckpt, args.id, str(args.num_round))
    print('Models are saved at %s' % (args.ckpt))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    if args.start_epoch > 1:
        args.resume_net1 = args.ckpt + '/' + 'net1_' + str(args.start_epoch) + '_checkpoint.pth.tar'
        args.resume_net2 = args.ckpt + '/' + 'net2_' + str(args.start_epoch) + '_checkpoint.pth.tar'

    args.running_lr = args.lr
    args.epoch_iters = math.ceil(int(len(train_dir)) / args.batch_size)
    args.max_iters = args.epoch_iters * args.num_epochs

    assert isinstance(args.crop_size, (int, list))
    if isinstance(args.crop_size, int):
        args.crop_size = [args.crop_size, args.crop_size, args.crop_size]

    assert isinstance(args.center_size, (int, list))
    if isinstance(args.center_size, int):
        args.center_size = [args.center_size, args.center_size, args.center_size]

    # save arguments to config file
    with open(args.ckpt + '/' + args.id + '.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    main(args)

