import torch
import torch.nn as nn
import numpy as np
from model.models import ModelBuilder
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import random
import nibabel as nib
from utils import AverageMeter
from distutils.version import LooseVersion
import argparse
from data_loader.dataset import ValDataset, norm
import SimpleITK as sitk

# save prediction results in the format of online submission
def visualize_result(name, pred, args):
    if not os.path.exists(args.result + '/' + str(args.num_round)):
        os.mkdir(args.result + '/' + str(args.num_round))
    pred = nib.Nifti1Image(pred, None)
    nib.save(pred, args.result + '/' + str(args.num_round) + '/' + str(name) + '.nii.gz')


# compute the number of segments of  the test images
def segment(image, mask, label, args):
    # find the left, right, bottom, top, forward, backward limit of the mask
    boundary = np.nonzero(mask)
    boundary = [np.unique(boundary[i]) for i in range(3)]
    limit = [(min(boundary[i]), max(boundary[i])) for i in range(3)]

    # compute the number of image segments in an image
    num_segments = [int(np.ceil((limit[i][1] - limit[i][0]) / args.center_size[i])) for i in range(3)]

    # compute the margin and the shape of the padding image
    margin = [args.crop_size[i] - args.center_size[i] for i in range(3)]
    padding_image_shape = [num_segments[i] * args.center_size[i] + margin[i] for i in range(3)]

    # start_index is corresponding to the location in the new padding images
    start_index = [limit[i][0] - int(margin[i] / 2) for i in range(3)]
    start_index = [int(-index) if index < 0 else 0 for index in start_index]

    # start and end is corresponding to the location in the original images
    start = [int(max(limit[i][0] - int(margin[i] / 2), 0)) for i in range(3)]
    end = [int(min(start[i] + padding_image_shape[i], mask.shape[i])) for i in range(3)]

    # compute the end_index corresponding to the new padding images
    end_index = [int(start_index[i] + end[i] - start[i]) for i in range(3)]

    # initialize the padding images
    # size = [start_index[i] if start_index[i] > 0 and end[i] < mask.shape[i] else 0 for i in range(3)] # change this code to solve 'can't broadcast' problem
    size = [start_index[i] if start_index[i] > 0  else 0 for i in range(3)]

    if sum(size) != 0:
        padding_image_shape = [sum(x) for x in zip(padding_image_shape, size)]

    mask_pad = np.zeros(padding_image_shape)
    label_pad = np.zeros(padding_image_shape)
    padding_image_shape.insert(0, args.num_input - 1)
    image_pad = np.zeros(padding_image_shape)


    # assign the original images to the padding images
    image_pad[:, start_index[0]: end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]] = image[:,
                                                                                                             start[0]:
                                                                                                             end[0],
                                                                                                             start[1]:
                                                                                                             end[1],
                                                                                                             start[2]:
                                                                                                             end[2]]
    label_pad[start_index[0]: end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]] = label[
                                                                                                          start[0]: end[
                                                                                                              0],
                                                                                                          start[1]:end[
                                                                                                              1],
                                                                                                          start[2]:end[
                                                                                                              2]]
    mask_pad[start_index[0]: end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]] = mask[
                                                                                                         start[0]: end[
                                                                                                             0],
                                                                                                         start[1]:end[
                                                                                                             1],
                                                                                                         start[2]:end[
                                                                                                             2]]
    return image_pad, mask_pad, label_pad, num_segments, (start_index, end_index), (start, end)


def accuracy(pred, mask, label):
    # columns in score is (# pred, # label, pred and label)
    score = np.zeros([3, 3])

    # compute Enhance score (label==4) in the first line
    score[0, 0] = np.count_nonzero(pred * mask == 4)
    score[0, 1] = np.count_nonzero(label == 4)
    score[0, 2] = np.count_nonzero(pred * mask * label == 16)

    # compute Core score (label == 1,3,4) in the second line
    pred[pred > 2] = 1
    label[label > 2] = 1
    score[1, 0] = np.count_nonzero(pred * mask == 1)
    score[1, 1] = np.count_nonzero(label == 1)
    score[1, 2] = np.count_nonzero(pred * mask * label == 1)

    # compute Whole score (all labels) in the third line
    pred[pred > 1] = 1
    label[label > 1] = 1
    score[2, 0] = np.count_nonzero(pred * mask == 1)
    score[2, 1] = np.count_nonzero(label == 1)
    score[2, 2] = np.count_nonzero(pred * mask * label == 1)
    return score

def test(test_loader, net1, net2, num_segments, args):
    # switch to evaluate mode
    net1.eval()
    net2.eval()

    # columns in score is (# pred, # label, pred and label)
    # lines in score is (Enhance, Core, Whole)
    score_net1, score_net2, score_avg = np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3])

    h_c, w_c, d_c = args.center_size
    pred_seg_net1 = np.zeros([num_segments[0] * h_c, num_segments[1] * w_c, d_c, args.num_classes])
    pred_seg_net2 = np.zeros([num_segments[0] * h_c, num_segments[1] * w_c, d_c, args.num_classes])
    pred_seg_avg = np.zeros([num_segments[0] * h_c, num_segments[1] * w_c, d_c, args.num_classes])

    for i, sample in enumerate(test_loader):
        image = sample['images'].float().cuda()
        target = sample['labels'].long().cuda()
        mask = sample['mask'].long().cuda()

        image = torch.squeeze(image, 0)
        target = torch.squeeze(target, 0)
        mask = torch.squeeze(mask, 0)

        with torch.no_grad():
            image = Variable(image)
            label = Variable(target)
            mask = Variable(mask)

            # The dimension of out should be in the dimension of B,C,H,W,D
            out1 = net1(image)
            out2 = net2(image)

            if args.net1 == 'Unet':
                crop_center_index = int((args.crop_size[0] - args.center_size[0]) // 2)
                out1 = out1[:, :, crop_center_index:-crop_center_index, crop_center_index:-crop_center_index,
                       crop_center_index:-crop_center_index]

            if args.net2 == 'Unet':
                crop_center_index = int((args.crop_size[0] - args.center_size[0]) // 2)
                out2 = out2[:, :, crop_center_index:-crop_center_index, crop_center_index:-crop_center_index,
                       crop_center_index:-crop_center_index]

            out_size = out1.size()[2:]
            out1 = out1.permute(0, 2, 3, 4, 1).contiguous().cuda()
            out2 = out2.permute(0, 2, 3, 4, 1).contiguous().cuda()
            out_avg = (out1 + out2)/2.0

            out1_data = (out1.data).cpu().numpy()
            out2_data = (out2.data).cpu().numpy()
            out_avg_data = (out_avg.data).cpu().numpy()

            # make the prediction
            out1 = out1.view(-1, args.num_classes).cuda()
            prediction1 = torch.max(out1, 1)[1].cuda().data.squeeze()
            out2 = out2.view(-1, args.num_classes).cuda()
            prediction2 = torch.max(out2, 1)[1].cuda().data.squeeze()
            out_avg = out_avg.view(-1, args.num_classes).cuda()
            prediction_avg = torch.max(out_avg, 1)[1].cuda().data.squeeze()

            # extract the center part of the label and mask
            start = [int((args.crop_size[k] - out_size[k]) / 2) for k in range(3)]
            end = [sum(x) for x in zip(start, out_size)]
            label = label[:, start[0]: end[0], start[1]: end[1], start[2]: end[2]]
            label = label.contiguous().view(-1)
            mask = mask[:, start[0]: end[0], start[1]: end[1], start[2]: end[2]]
            mask = mask.contiguous().view(-1)

        for j in range(num_segments[0]):
            pred_seg_net1[j * h_c:(j + 1) * h_c, i * d_c: (i + 1) * d_c, :, :] = out1_data[j, :]
            pred_seg_net2[j * h_c:(j + 1) * h_c, i * d_c: (i + 1) * d_c, :, :] = out2_data[j, :]
            pred_seg_avg[j * h_c:(j + 1) * h_c, i * d_c: (i + 1) * d_c, :, :] = out_avg_data[j, :]


            # compute the dice score
        score_net1 += accuracy(prediction1.data.cpu().numpy(), mask.data.cpu().numpy(), label.data.cpu().numpy())
        score_net2 += accuracy(prediction2.data.cpu().numpy(), mask.data.cpu().numpy(), label.data.cpu().numpy())
        score_avg += accuracy(prediction_avg.data.cpu().numpy(), mask.data.cpu().numpy(), label.data.cpu().numpy())


    return score_net1, score_net2, score_avg, pred_seg_net1, pred_seg_net2, pred_seg_avg


def main(args):
    # logging.info("round {}".format(args.num_round))
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
    net1 = torch.nn.DataParallel(net1, device_ids=list(range(args.num_gpus))).cuda()
    net2 = torch.nn.DataParallel(net2, device_ids=list(range(args.num_gpus))).cuda()
    cudnn.benchmark = True

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

            # initialization
    num_ignore = 0
    margin = [args.crop_size[k] - args.center_size[k] for k in range(3)]
    num_images = int(len(test_dir))
    dice_score_net1, dice_score_net2, dice_score_avg = np.zeros([num_images, 3]).astype(float), np.zeros([num_images, 3]).astype(float), np.zeros([num_images, 3]).astype(float)


    for i in range(num_images):
        # load the images, label and mask
        direct, _ = test_dir[i].split("\n")
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
        score_net1_per_image, score_net2_per_image, score_avg_per_image  = np.zeros([3, 3]), np.zeros([3, 3]), np.zeros([3, 3])
        # iterate over the z dimension
        for idz in range(num_segments[2]):
            tf = ValDataset(image_pad, label_pad, mask_pad, num_segments, idz, args)
            test_loader = DataLoader(tf, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers,
                                     pin_memory=False)
            score_net1_seg, score_net2_seg, score_avg_seg, pred_seg_net1, pred_seg_net2, pred_seg_avg =\
                test(test_loader, net1, net2, num_segments, args)
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
        pred_net1[index[0][0]:index[1][0], index[0][1]:index[1][1], index[0][2]:index[1][2]] = pred_net1_pad[:dist[0], :dist[1],
                                                                                          :dist[2]]
        pred_net2[index[0][0]:index[1][0], index[0][1]:index[1][1], index[0][2]:index[1][2]] = pred_net2_pad[:dist[0],
                                                                                               :dist[1],
                                                                                               :dist[2]]
        pred_avg[index[0][0]:index[1][0], index[0][1]:index[1][1], index[0][2]:index[1][2]] = pred_avg_pad[:dist[0],
                                                                                               :dist[1],
                                                                                               :dist[2]]


        if np.sum(score_net1_per_image[0, :]) == 0 or np.sum(score_net1_per_image[1, :]) == 0 or np.sum(
                score_net1_per_image[2, :]) == 0 or np.sum(score_net2_per_image[0, :]) == 0 or np.sum(
            score_net2_per_image[1, :])==0 or np.sum(score_net2_per_image[2, :]) == 0:
            num_ignore += 1
            continue

        # compute the Enhance, Core and Whole dice score
        net1_dice_score_per = [
            2 * np.sum(score_net1_per_image[k, 2]) / (np.sum(score_net1_per_image[k, 0]) + np.sum(score_net1_per_image[k, 1])) for k in
            range(3)]
        net2_dice_score_per = [
            2 * np.sum(score_net2_per_image[k, 2]) / (np.sum(score_net2_per_image[k, 0]) + np.sum(score_net2_per_image[k, 1])) for k in
            range(3)]
        avg_dice_score_per = [
            2 * np.sum(score_avg_per_image[k, 2]) / (np.sum(score_avg_per_image[k, 0]) + np.sum(score_avg_per_image[k, 1])) for k in
            range(3)]

        print('Image: %d,|| net1, Enhance: %.4f, Core: %.4f, Whole: %.4f || '
              'net2, Enhance: %.4f, Core: %.4f, Whole: %.4f || '
              'ensemble, Enhance: %.4f, Core: %.4f, Whole: %.4f' % (
        i, net1_dice_score_per[0], net1_dice_score_per[1], net1_dice_score_per[2],
        net2_dice_score_per[0], net2_dice_score_per[1], net2_dice_score_per[2],
        avg_dice_score_per[0], net2_dice_score_per[1], net2_dice_score_per[2]))

        dice_score_net1[i-num_ignore, :] = net1_dice_score_per
        dice_score_net2[i-num_ignore, :] = net2_dice_score_per
        dice_score_avg[i-num_ignore, :] = avg_dice_score_per

        if args.visualize:
            vis_net1 = np.argmax(pred_net1, axis=3)
            vis_net1 = vis_net1.transpose(1, 0, 2) # transpose for better vision
            visualize_result(patient_ID+"_net1", vis_net1, args)

            if vis_net1.any() > 0:
                print("-"*40)

            vis_net2 = np.argmax(pred_net2, axis=3)
            vis_net2 = vis_net2.transpose(1, 0, 2)  # transpose for better vision
            visualize_result(patient_ID + "_net2", vis_net2, args)

            vis_avg = np.argmax(pred_avg, axis=3)
            vis_avg = vis_avg.transpose(1, 0, 2)  # transpose for better vision
            visualize_result(patient_ID + "_avg", vis_avg, args)

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
    print('Evalution Done!')
    print("test epoch is ", args.test_epoch)
    print('net1||Enhance score: %.4f, Core score: %.4f, Whole score: %.4f, Mean Dice score: %.4f' % (
        mean_net1_dice[0], mean_net1_dice[1], mean_net1_dice[2], np.mean(mean_net1_dice)))
    print('net2||Enhance score: %.4f, Core score: %.4f, Whole score: %.4f, Mean Dice score: %.4f' % (
        mean_net2_dice[0], mean_net2_dice[1], mean_net2_dice[2], np.mean(mean_net2_dice)))
    print('ensemble||Enhance score: %.4f, Core score: %.4f, Whole score: %.4f, Mean Dice score: %.4f' % (
        mean_ensemble_dice[0], mean_ensemble_dice[1], mean_ensemble_dice[2], np.mean(mean_ensemble_dice)))

 
    # logging.info("test epoch is {}".format(args.test_epoch))
    # logging.info('net1||Enhance score: %.4f, Core score: %.4f, Whole score: %.4f, Mean Dice score: %.4f' % (
    #     mean_net1_dice[0], mean_net1_dice[1], mean_net1_dice[2], np.mean(mean_net1_dice)))
    # logging.info('net2||Enhance score: %.4f, Core score: %.4f, Whole score: %.4f, Mean Dice score: %.4f' % (
    #     mean_net2_dice[0], mean_net2_dice[1], mean_net2_dice[2], np.mean(mean_net2_dice)))
    # logging.info('ensemble||Enhance score: %.4f, Core score: %.4f, Whole score: %.4f, Mean Dice score: %.4f' % (
    #     mean_ensemble_dice[0], mean_ensemble_dice[1], mean_ensemble_dice[2], np.mean(mean_ensemble_dice)))

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.3.0'), \
        'PyTorch>=0.3.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='MEL',
                        help='a name for identitying the architecture.')
    parser.add_argument('--net1', default='Unet',
                        help='net1 in MEL')
    parser.add_argument('--net2', default='Unet',
                        help='net2 in MEL')

    # Path related arguments
    parser.add_argument('--test_path', default='datalist/test.txt',
                        help='txt file of the name of test data')
    parser.add_argument('--root_path',
                        default='/hjy/Dataset/MICCAI_BraTS_2018_Data_Training',
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
                        help='number of input image for each patient plus the mask')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--normalization', default=True, type=bool,
                        help='normalizae the data before running the test')
    parser.add_argument('--shuffle', default=False, type=bool,
                        help='if shuffle the data in test')
    parser.add_argument('--mask', default=True, type=bool,
                        help='if have the mask')

    # test related arguments
    parser.add_argument('--num_gpus', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='test batch size')
    parser.add_argument('--test_epoch', default=400, type=int,
                        help='epoch to start test.')
    parser.add_argument('--visualize', action='store_true',
                        help='save the prediction result as 3D images')
    parser.add_argument('--result', default='./result',
                        help='folder to output prediction results')
    parser.add_argument('--num_round', default=None, type=int,
                        help='restore the models from which run')
    parser.add_argument('--correction', dest='correction', type=bool, default=False)

    args = parser.parse_args()
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    test_file = open(args.test_path, 'r')
    test_dir = test_file.readlines()

    if not args.num_round:
        args.ckpt = os.path.join(args.ckpt, args.id)
    else:
        args.ckpt = os.path.join(args.ckpt, args.id, str(args.num_round))

    args.result = os.path.join(args.result, args.id)
    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    assert isinstance(args.crop_size, (int, list))
    if isinstance(args.crop_size, int):
        args.crop_size = [args.crop_size, args.crop_size, args.crop_size]

    assert isinstance(args.center_size, (int, list))
    if isinstance(args.center_size, int):
        args.center_size = [args.center_size, args.center_size, args.center_size]

    # do the test on a series of models
    args.resume_net1 = args.ckpt + '/' +'net1_'+ str(args.test_epoch) + '_checkpoint.pth.tar'
    args.resume_net2 = args.ckpt + '/' +'net2_'+ str(args.test_epoch) + '_checkpoint.pth.tar'
    main(args)













