import medpy.metric.binary as medpyMetrics
import numpy as np
import math
import torch
import torch.nn as nn


#
def toOrignalCategoryOneHot(labels):
    shape = labels.shape
    out = torch.zeros([shape[0], shape[1], shape[2], shape[3], 5])
    for i in range(5):
        out[:, :, :, :, i] = (labels == i)

    out = out.permute(0, 4, 1, 2, 3)
    return out

def softDice(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum(dim=(2, 3, 4))
    if nonSquared:
        union = (pred).sum() + (target).sum()
    else:
        union = (pred * pred).sum(dim=(2, 3, 4)) + (target * target).sum(dim=(2, 3, 4))
    dice = (2 * intersection + smoothing) / (union + smoothing)

    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])

    return dice.mean()

def dice(pred, target):
    predBin = (pred > 0.5).float()
    return softDice(predBin, target, 0, True).item()

def diceLoss(pred, target, nonSquared=False):
    return 1 - softDice(pred, target, nonSquared=nonSquared)

def bratsDiceLoss(outputs, labels, nonSquared=False):

    #bring outputs into correct shape
    wt, tc, et = outputs.chunk(3, dim=1)
    s = wt.shape
    wt = wt.view(s[0], s[2], s[3], s[4])
    tc = tc.view(s[0], s[2], s[3], s[4])
    et = et.view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask, tcMask, etMask = labels.chunk(3, dim=1)
    s = wtMask.shape
    wtMask = wtMask.view(s[0], s[2], s[3], s[4])
    tcMask = tcMask.view(s[0], s[2], s[3], s[4])
    etMask = etMask.view(s[0], s[2], s[3], s[4])

    #calculate losses
    wtLoss = diceLoss(wt, wtMask, nonSquared=nonSquared)
    tcLoss = diceLoss(tc, tcMask, nonSquared=nonSquared)
    etLoss = diceLoss(et, etMask, nonSquared=nonSquared)
    return (wtLoss + tcLoss + etLoss) / 5

def bratsDiceLossOriginal5(outputs, labels, nonSquared=False):
    outputList = list(outputs.chunk(5, dim=1))
    labels = toOrignalCategoryOneHot(labels)
    labelsList = list(labels.chunk(5, dim=1))
    totalLoss = 0
    for pred, target in zip(outputList, labelsList):
        pred = pred.cuda()
        target = target.cuda()
        # print("pred shape", pred.shape, print("target shape", target.shape))
        totalLoss = totalLoss + diceLoss(pred, target, nonSquared=nonSquared)
    return totalLoss.cuda()

def CrossEntropyDiceLoss(outputs, labels, nonSqured=False):
    CrossEntropyLoss = nn.CrossEntropyLoss(outputs, labels)
    DiceLoss = bratsDiceLossOriginal5(outputs, labels)
    totalLoss = CrossEntropyLoss + DiceLoss
    return totalLoss.cuda()

def ConsensusDiceLoss(pred1, pred2, target, smoothing=1, nonSquared=False):
    intersection = (pred1 * pred2 * target).sum(dim=(2, 3, 4))
    if nonSquared:
        union = (pred1).sum() + (pred2).sum() + (target).sum()
    else:
        union = (pred1 * pred1).sum(dim=(2, 3, 4)) + (pred2 * pred2).sum(dim=(2, 3, 4)) + (target * target).sum(dim=(2, 3, 4))
    consensusDice = (3 * intersection + smoothing) / (union + smoothing)

    # fix nans
    consensusDice[consensusDice != consensusDice] = consensusDice.new_tensor([1.0])

    return 1 - consensusDice.mean()


# This function is used for multual leraing for brain tumor segmentation
def bratsConsensusDiceLoss(outputs1, outputs2, labels, nonSquared=False):
    output1List = list(outputs1.chunk(5, dim=1))
    output2List = list(outputs2.chunk(5, dim=1))
    labels = toOrignalCategoryOneHot(labels)
    labelsList = list(labels.chunk(5, dim=1))
    totalLoss = 0
    for pred1, pred2, target in zip(output1List, output2List, labelsList):
        pred1 = pred1.cuda()
        pred2 = pred2.cuda()
        target = target.cuda()
        # print("pred shape", pred.shape, print("target shape", target.shape))
        totalLoss = totalLoss + ConsensusDiceLoss(pred1, pred2, target, nonSquared=nonSquared)
    return totalLoss.cuda()


def sensitivity(pred, target):
    predBin = (pred > 0.5).float()
    intersection = (predBin * target).sum()
    allPositive = target.sum()

    # special case for zero positives
    if allPositive == 0:
        return 1.0
    return (intersection / allPositive).item()

def specificity(pred, target):
    predBinInv = (pred <= 0.5).float()
    targetInv = (target == 0).float()
    intersection = (predBinInv * targetInv).sum()
    allNegative = targetInv.sum()
    return (intersection / allNegative).item()

def getHd95(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    if np.count_nonzero(pred) > 0 and np.count_nonzero(target):
        surDist1 = medpyMetrics.__surface_distances(pred, target)
        surDist2 = medpyMetrics.__surface_distances(target, pred)
        hd95 = np.percentile(np.hstack((surDist1, surDist2)), 95)
        return hd95
    else:
        # Edge cases that medpy cannot handle
        return -1

def getWTMask(labels):
    return (labels != 0).float()

def getTCMask(labels):
    return ((labels != 0) * (labels != 2)).float() #We use multiplication as AND

def getETMask(labels):
    return (labels == 4).float()

def predictionOverlapLoss(pred1, pred2, smoothing=1, nonSquared=False):
    intersection = (pred1 * pred2).sum(dim=(2, 3, 4))
    if nonSquared:
        union = (pred1).sum() + (pred2).sum()
    else:
        union = (pred1 * pred1).sum(dim=(2, 3, 4)) + (pred2 * pred2).sum(dim=(2, 3, 4))
    predictionOverlap = (2 * intersection + smoothing) / (union + smoothing)

    # fix nans
    predictionOverlap[predictionOverlap != predictionOverlap] = predictionOverlap.new_tensor([1.0])

    return 1 - predictionOverlap.mean()



def bratsPredictionOverlap(outputs1, outputs2, nonSquared=False):
    output1List = list(outputs1.chunk(5, dim=1))
    output2List = list(outputs2.chunk(5, dim=1))
    totalLoss = 0
    for pred1, pred2 in zip(output1List, output2List):
        pred1 = pred1.cuda()
        pred2 = pred2.cuda()
        # print("pred shape", pred.shape, print("target shape", target.shape))
        totalLoss = totalLoss + predictionOverlapLoss(pred1, pred2, nonSquared=nonSquared)
    return totalLoss.cuda()


if __name__ == '__main__':
    outputs1 = torch.rand((10, 5, 25, 25, 25)).cuda()
    outputs2 = torch.rand((10, 5, 25, 25, 25)).cuda()
    labels = torch.randint(0, 6, (10, 25, 25, 25)).cuda()
    print(labels.shape)
    # loss = bratsConsensusDiceLoss(outputs1, outputs2, labels)
    # print(loss)


    # labels = toOrignalCategoryOneHot(labels)
    # output_list = list(outputs.chunk(5, dim=1))
    # print(len(output_list), output_list)
    # print(labels.shape)
    loss = bratsDiceLossOriginal5(outputs1, labels)
    o = outputs1.sum(dim=(2, 3, 4))
    print(loss)
    print(o.shape)

