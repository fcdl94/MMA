import argparse
import os
import datetime
import logging
import time
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import numpy as np

from maskrcnn_benchmark.modeling.rpn.utils import permute_and_flatten
from maskrcnn_benchmark.layers import smooth_l1_loss

def calculate_rpn_distillation_loss(rpn_output_source, rpn_output_target, cls_loss=None, bbox_loss=None, bbox_threshold=None):

    rpn_objectness_source, rpn_bbox_regression_source = rpn_output_source
    rpn_objectness_target, rpn_bbox_regression_target = rpn_output_target

    # calculate rpn classification loss
    num_source_rpn_objectness = len(rpn_objectness_source)
    num_target_rpn_objectness = len(rpn_objectness_target)
    final_rpn_cls_distillation_loss = []
    objectness_difference = []

    if num_source_rpn_objectness == num_target_rpn_objectness:
        for i in range(num_target_rpn_objectness):
            current_source_rpn_objectness = rpn_objectness_source[i]
            current_target_rpn_objectness = rpn_objectness_target[i]
            if cls_loss == 'filtered_l1':
                rpn_objectness_difference = current_source_rpn_objectness - current_target_rpn_objectness
                objectness_difference.append(rpn_objectness_difference)
                filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                rpn_distillation_loss = torch.max(rpn_objectness_difference, filter)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            elif cls_loss == 'filtered_l2':
                rpn_objectness_difference = current_source_rpn_objectness - current_target_rpn_objectness
                objectness_difference.append(rpn_objectness_difference)
                filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                rpn_difference = torch.max(rpn_objectness_difference, filter)
                rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            elif cls_loss == 'normalized_filtered_l2':
                avrage_source_rpn_objectness = torch.mean(current_source_rpn_objectness)
                average_target_rpn_objectness = torch.mean(current_target_rpn_objectness)
                normalized_source_rpn_objectness = current_source_rpn_objectness - avrage_source_rpn_objectness
                normalized_target_rpn_objectness = current_target_rpn_objectness - average_target_rpn_objectness
                rpn_objectness_difference = normalized_source_rpn_objectness - normalized_target_rpn_objectness
                objectness_difference.append(rpn_objectness_difference)
                filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                rpn_difference = torch.max(rpn_objectness_difference, filter)
                rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            elif cls_loss == 'masked_filtered_l2':
                source_mask = current_source_rpn_objectness.clone()
                source_mask[current_source_rpn_objectness >= 0.7] = 1  # rpn threshold for foreground
                source_mask[current_source_rpn_objectness < 0.7] = 0
                rpn_objectness_difference = current_source_rpn_objectness - current_target_rpn_objectness
                masked_rpn_objectness_difference = rpn_objectness_difference * source_mask
                objectness_difference.append(masked_rpn_objectness_difference)
                filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                rpn_difference = torch.max(masked_rpn_objectness_difference, filter)
                rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            else:
                raise ValueError("Wrong loss function for rpn classification distillation")
    else:
        raise ValueError("Wrong rpn objectness output")
    final_rpn_cls_distillation_loss = sum(final_rpn_cls_distillation_loss)/num_source_rpn_objectness
    #a = objectness_difference > 0

    # calculate rpn bounding box regression loss
    num_source_rpn_bbox = len(rpn_bbox_regression_source)
    num_target_rpn_bbox = len(rpn_bbox_regression_target)
    final_rpn_bbs_distillation_loss = []
    l2_loss = nn.MSELoss(size_average=False, reduce=False)

    if num_source_rpn_bbox == num_target_rpn_bbox:
        for i in range(num_target_rpn_bbox):
            current_source_rpn_bbox = rpn_bbox_regression_source[i]
            current_target_rpn_bbox = rpn_bbox_regression_target[i]
            current_objectness_difference = objectness_difference[i]
            [N, A, H, W] = current_objectness_difference.size()  # second dimention contains location shifting information for each anchor
            current_objectness_difference = permute_and_flatten(current_objectness_difference, N, A, 1, H, W)
            current_source_rpn_bbox = permute_and_flatten(current_source_rpn_bbox, N, A, 4, H, W)
            current_target_rpn_bbox = permute_and_flatten(current_target_rpn_bbox, N, A, 4, H, W)
            current_objectness_mask = current_objectness_difference.clone()
            current_objectness_mask[current_objectness_difference > bbox_threshold] = 1
            current_objectness_mask[current_objectness_difference <= bbox_threshold] = 0
            masked_source_rpn_bbox = current_source_rpn_bbox * current_objectness_mask
            masked_target_rpn_bbox = current_target_rpn_bbox * current_objectness_mask
            if bbox_loss == 'l2':
                current_bbox_distillation_loss = l2_loss(masked_source_rpn_bbox, masked_target_rpn_bbox)
                final_rpn_bbs_distillation_loss.append(torch.mean(torch.mean(torch.sum(current_bbox_distillation_loss, dim=2), dim=1), dim=0))
            elif bbox_loss == 'l1':
                current_bbox_distillation_loss = torch.abs(masked_source_rpn_bbox - masked_source_rpn_bbox)
                final_rpn_bbs_distillation_loss.append(torch.mean(torch.mean(torch.sum(current_bbox_distillation_loss, dim=2), dim=1), dim=0))
            elif bbox_loss == 'None':
                final_rpn_bbs_distillation_loss.append(0)
            else:
                raise ValueError('Wrong loss function for rpn bounding box regression distillation')
    else:
        raise ValueError('Wrong RPN bounding box regression output')
    final_rpn_bbs_distillation_loss = sum(final_rpn_bbs_distillation_loss)/num_source_rpn_bbox

    final_rpn_loss = final_rpn_cls_distillation_loss + final_rpn_bbs_distillation_loss
    final_rpn_loss.to('cuda')

    return final_rpn_loss


def calculate_roi_align_distillation(source_roi_align_features, target_roi_align_features):
    final_roi_align_distillation_loss = []

    l2_loss = nn.MSELoss(size_average=False, reduce=False)
    roi_align_distillation_loss = l2_loss(source_roi_align_features, target_roi_align_features)
    final_roi_align_distillation_loss.append(torch.mean(roi_align_distillation_loss))

    return sum(final_roi_align_distillation_loss)


def calculate_feature_distillation_loss(source_features, target_features, loss=None):  # pixel-wise
    num_source_features = len(source_features)
    num_target_features = len(target_features)
    final_feature_distillation_loss = []

    if num_source_features == num_target_features:
        for i in range(num_source_features):
            source_feature = source_features[i]
            target_feature = target_features[i]
            if loss == 'l2':
                l2_loss = nn.MSELoss(size_average=False, reduce=False)
                feature_distillation_loss = l2_loss(source_feature, target_feature)
                final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
            elif loss == 'l1':
                feature_distillation_loss = torch.abs(source_feature - target_feature)
                final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
            elif loss == 'smooth_l1':
                feature_distillation_loss = smooth_l1_loss(source_feature, target_feature, size_average=True, beta=1)
                final_feature_distillation_loss.append(feature_distillation_loss)
            elif loss == 'normalized_filtered_l1':
                source_feature_avg = torch.mean(source_feature)
                target_feature_avg = torch.mean(target_feature)
                normalized_source_feature = source_feature - source_feature_avg  # normalize features
                normalized_target_feature = target_feature - target_feature_avg
                feature_difference = normalized_source_feature - normalized_target_feature
                feature_size = feature_difference.size()
                filter = torch.zeros(feature_size).to('cuda')
                feature_distillation_loss = torch.max(feature_difference, filter)
                final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            elif loss == 'normalized_filtered_l2':
                source_feature_avg = torch.mean(source_feature)
                target_feature_avg = torch.mean(target_feature)
                normalized_source_feature = source_feature - source_feature_avg  # normalize features
                normalized_target_feature = target_feature - target_feature_avg  # normalize features
                feature_difference = normalized_source_feature - normalized_target_feature
                feature_size = feature_difference.size()
                filter = torch.zeros(feature_size).to('cuda')
                feature_distillation = torch.max(feature_difference, filter)
                feature_distillation_loss = torch.mul(feature_distillation, feature_distillation)
                final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            else:
                raise ValueError("Wrong loss function for feature distillation")
    else:
        raise ValueError("Number of source features must equal to number of target features")

    final_feature_distillation_loss = sum(final_feature_distillation_loss)

    return final_feature_distillation_loss


def calculate_roi_distillation_loss(soften_results, target_results, cls_preprocess=None, cls_loss=None, bbs_loss=None, temperature=1):

    soften_scores, soften_bboxes = soften_results
    target_scores, target_bboxes = target_results
    num_of_distillation_categories = soften_scores.size()[1]

    # compute distillation loss
    if cls_preprocess == 'sigmoid':
        soften_scores = F.sigmoid(soften_scores)
        target_scores = F.sigmoid(target_scores)
        modified_soften_scores = soften_scores[:, : num_of_distillation_categories]  # include background
        modified_target_scores = target_scores[:, : num_of_distillation_categories]  # include background
    elif cls_preprocess == 'softmax':  # exp(x_i) / exp(x).sum()
        modified_soften_scores = F.softmax(soften_scores, dim=1)
        modified_target_scores = F.log_softmax(target_scores[:, : num_of_distillation_categories], dim=1)
    elif cls_preprocess == 'softmax_all':  # exp(x_i) / exp(x).sum()
        modified_soften_scores = F.softmax(soften_scores, dim=1)
        modified_target_scores = F.log_softmax(target_scores, dim=1)[:, : num_of_distillation_categories]
    elif cls_preprocess == 'log_softmax':  # log( exp(x_i) / exp(x).sum() )
        soften_scores = F.log_softmax(soften_scores)
        target_scores = F.log_softmax(target_scores)
        modified_soften_scores = soften_scores[:, : num_of_distillation_categories]  # include background
        modified_target_scores = target_scores[:, : num_of_distillation_categories]  # include background
    elif cls_preprocess == 'normalization':
        class_wise_soften_scores_avg = torch.mean(soften_scores, dim=1).view(-1, 1)
        class_wise_target_scores_avg = torch.mean(target_scores, dim=1).view(-1, 1)
        normalized_soften_scores = torch.sub(soften_scores, class_wise_soften_scores_avg)
        normalized_target_scores = torch.sub(target_scores, class_wise_target_scores_avg)
        modified_soften_scores = normalized_target_scores[:, : num_of_distillation_categories]  # include background
        modified_target_scores = normalized_soften_scores[:, : num_of_distillation_categories]  # include background
    elif cls_preprocess == 'raw':
        modified_soften_scores = soften_scores[:, : num_of_distillation_categories]  # include background
        modified_target_scores = target_scores[:, : num_of_distillation_categories]  # include background
    elif cls_preprocess == 'none':  # FOR UNBIAS CROSS ENTROPY USE THIS
        modified_soften_scores = soften_scores
        modified_target_scores = target_scores
    else:
        raise ValueError("Wrong preprocessing method for raw classification output")

    tot_classes = target_scores.size()[1]
    if cls_loss == 'l2':
        l2_loss = nn.MSELoss(size_average=False, reduce=False)
        class_distillation_loss = l2_loss(modified_soften_scores, modified_target_scores)
        class_distillation_loss = torch.mean(torch.mean(class_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
    elif cls_loss == 'cross-entropy':  # softmax/sigmoid + cross-entropy
        class_distillation_loss = - modified_soften_scores * modified_target_scores
        class_distillation_loss = torch.mean(torch.mean(class_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
    elif cls_loss == 'cross-entropy-sum':  # softmax/sigmoid + cross-entropy
        class_distillation_loss = - modified_soften_scores * modified_target_scores
        class_distillation_loss = torch.mean(torch.sum(class_distillation_loss, dim=1))  # average towards categories and proposals
    elif cls_loss == 'unbiased-cross-entropy':
        new_bkg_idx = torch.tensor([0] + [x for x in range(
            num_of_distillation_categories, tot_classes)]).to(target_scores.device)
        den = torch.logsumexp(modified_target_scores, dim=1)
        outputs_no_bgk = modified_target_scores[:, 1:-(tot_classes-num_of_distillation_categories)] - den.unsqueeze(dim=1)
        outputs_bkg = torch.logsumexp(torch.index_select(modified_target_scores, index=new_bkg_idx, dim=1), dim=1) - den
        labels = torch.softmax(modified_soften_scores, dim=1)
        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / soften_scores.shape[1]
        class_distillation_loss = -torch.mean(loss)
    elif cls_loss == 'softmax cross-entropy with temperature':  # raw + softmax cross-entropy with temperature
        log_softmax = nn.LogSoftmax()
        softmax = nn.Softmax()
        class_distillation_loss = - softmax(modified_soften_scores/temperature) * log_softmax(modified_target_scores/temperature)
        class_distillation_loss = class_distillation_loss * temperature * temperature
        class_distillation_loss = torch.mean(torch.mean(class_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
    elif cls_loss == 'filtered_l2':
        cls_difference = modified_soften_scores - modified_target_scores
        filter = torch.zeros(modified_soften_scores.size()).to('cuda')
        class_distillation_loss = torch.max(cls_difference, filter)
        class_distillation_loss = class_distillation_loss * class_distillation_loss
        class_distillation_loss = torch.mean(torch.mean(class_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
        del filter
        torch.cuda.empty_cache()  # Release unoccupied memory
    else:
        raise ValueError("Wrong loss function for classification")

    # compute distillation bbox loss
    modified_soften_boxes = soften_bboxes[:, 1:, :]  # exclude background bbox
    modified_target_bboxes = target_bboxes[:, 1:num_of_distillation_categories, :]  # exclude background bbox
    if bbs_loss == 'l2':
        l2_loss = nn.MSELoss(size_average=False, reduce=False)
        bbox_distillation_loss = l2_loss(modified_target_bboxes, modified_soften_boxes)
        bbox_distillation_loss = torch.mean(torch.mean(torch.sum(bbox_distillation_loss, dim=2), dim=1), dim=0)  # average towards categories and proposals
    elif bbs_loss == 'smooth_l1':
        num_bboxes = modified_target_bboxes.size()[0]
        num_categories = modified_target_bboxes.size()[1]
        bbox_distillation_loss = smooth_l1_loss(modified_target_bboxes, modified_soften_boxes, size_average=False, beta=1)
        bbox_distillation_loss = bbox_distillation_loss / (num_bboxes * num_categories)  # average towards categories and proposals
    else:
        raise ValueError("Wrong loss function for bounding box regression")

    roi_distillation_losses = torch.add(class_distillation_loss, bbox_distillation_loss)

    return roi_distillation_losses


def calculate_roi_distillation_losses(soften_results, target_results, dist='l2'):

    if dist == 'ce':
        cls_preprocess = 'softmax'
        cls_loss = 'cross-entropy'
        bbs_loss = 'l2'
        temperature = 1
    elif dist == 'ce_ada':
        cls_preprocess = 'softmax'
        cls_loss = 'cross-entropy-sum'
        bbs_loss = 'l2'
        temperature = 1
    elif dist == 'ce_all':
        cls_preprocess = 'softmax_all'
        cls_loss = 'cross-entropy'
        bbs_loss = 'l2'
        temperature = 1
    elif dist == 'uce':
        cls_preprocess = 'none'
        cls_loss = 'unbiased-cross-entropy'
        bbs_loss = 'l2'
        temperature = 1
    else:
        cls_preprocess = 'normalization'
        cls_loss = 'l2'
        bbs_loss = 'l2'
        temperature = 1

    roi_distillation_losses = calculate_roi_distillation_loss(
        soften_results, target_results, cls_preprocess, cls_loss, bbs_loss, temperature)

    return roi_distillation_losses


def calculate_mask_distillation_losses(soften_mask_logits, target_mask_logits):
    num_of_distillation_categories = soften_mask_logits.shape[1]
    soften_mask_logits = torch.sigmoid(soften_mask_logits)
    old_classes_target_mask_logits = target_mask_logits[:, :num_of_distillation_categories]
    mask_distillation_loss = nn.functional.binary_cross_entropy_with_logits(old_classes_target_mask_logits, soften_mask_logits)
    return mask_distillation_loss


def calculate_roi_distillation_losses_old(model_source, model_target, images, gt_proposals=None, distributed=False, dist='l2'):

    # --- calculate roi-subnet classification and bbox regression distillation loss ---
    # do test on the pre-trained frozen source model to get the soften label
    with torch.no_grad():
        soften_result, soften_mask_logits, soften_proposal, feature_source, backbone_feature_source, anchor_source, rpn_output_source, roi_align_features_source = \
            model_source.generate_soften_proposal(images)

    # use soften proposal and soften result to calculate distillation loss
    # 'num_of_distillation_categories' = number of categories for source model including background
    if distributed:
        model_target = model_target.module

    if dist == 'ce':
        roi_distillation_losses, roi_align_features_target = model_target.calculate_roi_distillation_loss(
            images, soften_proposal, soften_result, cls_preprocess='softmax', cls_loss='cross-entropy', bbs_loss='l2', temperature=1)
    elif dist == 'uce':
        roi_distillation_losses, roi_align_features_target = model_target.calculate_roi_distillation_loss(
            images, soften_proposal, soften_result, cls_preprocess='raw', cls_loss='unbiased-cross-entropy', bbs_loss='l2', temperature=1)
    else:
        roi_distillation_losses, roi_align_features_target = model_target.calculate_roi_distillation_loss(
            images, soften_proposal, soften_result, cls_preprocess='normalization', cls_loss='l2', bbs_loss='l2', temperature=1)

    return roi_distillation_losses, rpn_output_source, feature_source, backbone_feature_source, \
           soften_result, soften_proposal, roi_align_features_source, roi_align_features_target




