import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.models import ModelBuilder
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import random
import time
import nibabel as nib
from utils import AverageMeter
from distutils.version import LooseVersion
import argparse
from data_loader.dataset import TestDataset
import SimpleITK as sitk


# save prediction results in the format of online submission
def visualize_result(name, pred, args):
    if not os.path.exists(args.result + '/' + str(args.num_round) + '/' + 'submission/'):
        os.makedirs(args.result + '/' + str(args.num_round) + '/' + 'submission/')
    pred = nib.Nifti1Image(pred, None)
    nib.save(pred, args.result + '/' + str(args.num_round) + '/' + 'submission/' + str(name) + '.nii.gz')


def norm(image):
    image = np.squeeze(image)
    image_nonzero = image[np.nonzero(image)]
    return (image - image_nonzero.mean()) / image_nonzero.std()


# compute the number of segments of  the test images
def segment(image, mask, args):
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
    size = [start_index[i] if start_index[i] > 0 and end[i] < mask.shape[i] else 0 for i in range(3)]
    if sum(size) != 0:
        padding_image_shape = [sum(x) for x in zip(padding_image_shape, size)]

    mask_pad = np.zeros(padding_image_shape)
    padding_image_shape.insert(0, args.num_input)
    image_pad = np.zeros(padding_image_shape)

    # assign the original images to the padding images
    image_pad[:, start_index[0]: end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]] = image[:,
                                                                                                             start[0]:
                                                                                                             end[0],
                                                                                                             start[1]:
                                                                                                             end[1],
                                                                                                             start[2]:
                                                                                                             end[2]]
    mask_pad[start_index[0]: end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]] = mask[
                                                                                                         start[0]: end[
                                                                                                             0],
                                                                                                         start[1]:end[
                                                                                                             1],
                                                                                                         start[2]:end[
                                                                                                             2]]
    return image_pad, mask_pad, num_segments, (start_index, end_index), (start, end)


def test(test_loader, net1, net2, num_segments, args):
    # switch to evaluate mode
    net1.eval()
    net2.eval()
    h_c, w_c, d_c = args.center_size
    pred_seg1 = np.zeros([num_segments[0] * h_c, num_segments[1] * w_c, d_c, args.num_classes])
    pred_seg2 = np.zeros([num_segments[0] * h_c, num_segments[1] * w_c, d_c, args.num_classes])

    for i, sample in enumerate(test_loader):
        image = sample['images'].float().cuda()
        image = torch.squeeze(image, 0)

        with torch.no_grad():
            image = Variable(image)
            if not args.use_TTA:
                # The dimension of out should be in the dimension of B,C,H,W,D
                out1 = net1(image)
                out2 = net2(image)
            else:
                out1 = F.softmax(net1(image), 1)  # 000
                out1 += F.softmax(net1(image.flip(dims=(2,))).flip(dims=(2,)), 1)
                out1 += F.softmax(net1(image.flip(dims=(3,))).flip(dims=(3,)), 1)
                out1 += F.softmax(net1(image.flip(dims=(4,))).flip(dims=(4,)), 1)
                out1 += F.softmax(net1(image.flip(dims=(2, 3))).flip(dims=(2, 3)), 1)
                out1 += F.softmax(net1(image.flip(dims=(2, 4))).flip(dims=(2, 4)), 1)
                out1 += F.softmax(net1(image.flip(dims=(3, 4))).flip(dims=(3, 4)), 1)
                out1 += F.softmax(net1(image.flip(dims=(2, 3, 4))).flip(dims=(2, 3, 4)), 1)
                out1 = out1 / 8.0  # mean

                out2 = F.softmax(net2(image), 1)  # 000
                out2 += F.softmax(net2(image.flip(dims=(2,))).flip(dims=(2,)), 1)
                out2 += F.softmax(net2(image.flip(dims=(3,))).flip(dims=(3,)), 1)
                out2 += F.softmax(net2(image.flip(dims=(4,))).flip(dims=(4,)), 1)
                out2 += F.softmax(net2(image.flip(dims=(2, 3))).flip(dims=(2, 3)), 1)
                out2 += F.softmax(net2(image.flip(dims=(2, 4))).flip(dims=(2, 4)), 1)
                out2 += F.softmax(net2(image.flip(dims=(3, 4))).flip(dims=(3, 4)), 1)
                out2 += F.softmax(net2(image.flip(dims=(2, 3, 4))).flip(dims=(2, 3, 4)), 1)
                out2 = out2 / 8.0  # mean
            if args.net1 == 'Unet':
                crop_center_index = int((args.crop_size[0] - args.center_size[0]) // 2)
                out1 = out1[:, :, crop_center_index:-crop_center_index, crop_center_index:-crop_center_index,
                       crop_center_index:-crop_center_index]

            if args.net2 == 'Unet':
                crop_center_index = int((args.crop_size[0] - args.center_size[0]) // 2)
                out2 = out2[:, :, crop_center_index:-crop_center_index, crop_center_index:-crop_center_index,
                       crop_center_index:-crop_center_index]

            out1_size = out1.size()[2:]
            out1 = out1.permute(0, 2, 3, 4, 1).contiguous().cuda()
            out2 = out2.permute(0, 2, 3, 4, 1).contiguous().cuda()

            out1_data = (out1.data).cpu().numpy()
            out2_data = (out2.data).cpu().numpy()

        for j in range(num_segments[0]):
            pred_seg1[j * h_c:(j + 1) * h_c, i * d_c: (i + 1) * d_c, :, :] = out1_data[j, :]
            pred_seg2[j * h_c:(j + 1) * h_c, i * d_c: (i + 1) * d_c, :, :] = out2_data[j, :]
    pred_seg = (pred_seg1 + pred_seg2)/2.0
    return pred_seg


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
    dice_score = np.zeros([num_images, 3]).astype(float)

    start = time.time()
    for i in range(num_images):
        # load the images, label and mask
        patient_ID, _ = test_dir[i].split("\n")

        if args.correction:
            flair = nib.load(
                os.path.join(args.root_path, patient_ID, patient_ID + '_flair_corrected.nii.gz')).get_data()
            t2 = nib.load(os.path.join(args.root_path, patient_ID, patient_ID + '_t2_corrected.nii.gz')).get_data()
            t1 = nib.load(os.path.join(args.root_path, patient_ID, patient_ID + '_t1_corrected.nii.gz')).get_data()
            t1ce = nib.load(os.path.join(args.root_path, patient_ID, patient_ID + '_t1ce_corrected.nii.gz')).get_data()
            # print('Using bias correction dataset')
        else:
            flair = nib.load(os.path.join(args.root_path, patient_ID, patient_ID + '_flair.nii.gz')).get_data()

            t2 = nib.load(os.path.join(args.root_path, patient_ID, patient_ID + '_t2.nii.gz')).get_data()

            t1 = nib.load(os.path.join(args.root_path, patient_ID, patient_ID + '_t1.nii.gz')).get_data()

            t1ce = nib.load(os.path.join(args.root_path, patient_ID, patient_ID + '_t1ce.nii.gz')).get_data()
            # print("not using bias correction correction dataset")

        mask = nib.load(os.path.join(args.root_path, patient_ID, patient_ID + '_mask.nii.gz')).get_data()
        mask = mask.astype(int)

        flair = np.expand_dims(norm(flair), axis=0).astype(float)
        t2 = np.expand_dims(norm(t2), axis=0).astype(float)
        t1 = np.expand_dims(norm(t1), axis=0).astype(float)
        t1ce = np.expand_dims(norm(t1ce), axis=0).astype(float)
        images = np.concatenate([flair, t2, t1, t1ce], axis=0).astype(float)

        # divide the input images input small image segments
        # return the padding input images which can be divided exactly
        image_pad, mask_pad, num_segments, padding_index, index = segment(images, mask, args)

        # initialize prediction for the whole image as background
        mask_shape = list(mask.shape)
        mask_shape.append(args.num_classes)
        pred = np.zeros(mask_shape)
        pred[:, :, :, 0] = 1

        # initialize the prediction for a small segmentation as background
        pad_shape = [int(num_segments[k] * args.center_size[k]) for k in range(3)]
        pad_shape.append(args.num_classes)
        pred_pad = np.zeros(pad_shape)
        pred_pad[:, :, :, 0] = 1

        # iterate over the z dimension
        for idz in range(num_segments[2]):
            tf = TestDataset(image_pad, mask_pad, num_segments, idz, args)
            test_loader = DataLoader(tf, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers,
                                     pin_memory=False)
            pred_seg = test(test_loader, net1, net2, num_segments, args)
            pred_pad[:, :, idz * args.center_size[2]:(idz + 1) * args.center_size[2], :] = pred_seg

            # decide the start and end point in the original image
        for k in range(3):
            if index[0][k] == 0:
                index[0][k] = int(margin[k] / 2 - padding_index[0][k])
            else:
                index[0][k] = int(margin[k] / 2 + index[0][k])

            index[1][k] = int(min(index[0][k] + num_segments[k] * args.center_size[k], mask.shape[k]))

        dist = [index[1][k] - index[0][k] for k in range(3)]
        pred[index[0][0]:index[1][0], index[0][1]:index[1][1], index[0][2]:index[1][2]] = pred_pad[:dist[0], :dist[1],
                                                                                          :dist[2]]

        if args.visualize:
            vis = np.argmax(pred, axis=3).astype('uint8')
            if args.postprocess:
                ET_voxels = (vis == 4).sum()
                if ET_voxels < 500:
                    vis[np.where(vis == 4)] = 1
            visualize_result(patient_ID, vis, args)
        print(patient_ID)
    print('Evalution Done!')
    print('Average time for each scan: {:.4f} seconds'.format((time.time() - start)/num_images))


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.3.0'), \
        'PyTorch>=0.3.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='MEL',
                        help='a name for identitying the architecture.')
    parser.add_argument('--net1', default='Unet',
                        help='a name for identitying the subnet. Choose from the following options: Basic, Unet.')
    parser.add_argument('--net2', default='Unet',
                        help='a name for identitying the subnet. Choose from the following options: Basic, Unet.')

    # Path related arguments
    parser.add_argument('--test_path', default='datalist/test_online.txt',
                        help='txt file of the name of test data')
    parser.add_argument('--root_path',
                        default='/hjy/Dataset/MICCAI_BraTS_2018_Data_Validation/',
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
    parser.add_argument('--num_input', default=4, type=int,
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
    parser.add_argument('--gpu', default='1', type=str, help='Supprot one GPU & multiple GPUs.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='test batch size')
    parser.add_argument('--test_epoch', default=400, type=int,
                        help='epoch to start test.')
    parser.add_argument('--visualize', type=bool, default=True,
                        help='save the prediction result as 3D images')
    parser.add_argument('--result', default='./result',
                        help='folder to output prediction results')
    parser.add_argument('--num_round', default=None, type=int,
                        help='restore the models from which run')
    parser.add_argument('--correction', dest='correction', type=bool, default=False)
    parser.add_argument('--use_TTA', type=bool, default=False,
                        help='test time augmentation')
    parser.add_argument('--postprocess', type=bool, default=False,
                        help='test time augmentation')

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













