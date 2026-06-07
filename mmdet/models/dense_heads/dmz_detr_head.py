# Copyright (c) OpenMMLab. All rights reserved.
# 加伪框all_pseudo_boxes
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, bbox2result, bbox_overlaps, multiclass_nms
from mmdet.models.utils.transformer import inverse_sigmoid
from ..builder import HEADS, build_loss
from .deformable_detr_head import DeformableDETRHead
from .detr_head import DETRHead

import os
import cv2
import numpy as np
import torch


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activations = nn.ModuleList(nn.GELU() for _ in range(num_layers - 1))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activations[i](layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
    

class Queues(nn.Module):
    def __init__(self, names, lengths, emb_dim=512):
        super(Queues, self).__init__()
        self.names = names
        self.lengths = lengths
        self.emb_dim = emb_dim
        self._init_queues()

    def _init_queues(self):
        attr_names = self.names
        queue_lengths = self.lengths
        for n in attr_names:
            self.register_buffer(n, torch.ones(0, self.emb_dim), persistent=False)
        self.queue_lengths = {n: queue_lengths[i] for i, n in enumerate(attr_names)}

    @torch.no_grad()
    def dequeue_and_enqueue(self, queue_update):
        for k, feat in queue_update.items():
            if len(feat) == 0:
                continue
            queue_length = self.queue_lengths[k]
            in_length = feat.shape[0]
            queue_value = getattr(self, k)
            current_length = queue_value.shape[0]
            kept_length = min(queue_length - in_length, current_length)

            queue_value.data = torch.cat([feat, queue_value[:kept_length]])

    @torch.no_grad()
    def get_queue(self, key):
        value = getattr(self, key)
        return value 



@HEADS.register_module()
class DmzDETRHead(DETRHead):
    """Head of DeformDETR: Deformable DETR: Deformable Transformers for End-to-
    End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(self,
                 *args,
                 with_box_refine=True,
                 as_two_stage=False,
                 transformer=None,
                 
                 num_stages=6,
                 stage_loss_weights=[1] * 6,
                 text_dim=512,
                 content_dim=256,
                 clip_model_path=None,
                 max_pseudo_box_num=5,
                 cls_tau=50,
                 skd_tau=20,
                 rkd_tau=5,
                 alpha=0.35,
                 beta=0.65,
                 pre_extracted_clip_text_feat=None,
                 use_pseudo_box=False,
                 split_visual_text=False,
                 use_text_space_rkd_loss=False,
                 loss_visual_skd=dict(type='CrossEntropyLoss', loss_weight=0.5),
                 loss_visual_rkd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=5.0),
                 use_image_level_distill=True,
                 num_additional_padding_prompts=32,
                 novel_obj_queue_dict=None,
                 
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage

        super(DmzDETRHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        # # train_cfg would be None when run the test.py
        # if train_cfg is not None:
        #     for stage in range(num_stages):
        #         assert isinstance(self.bbox_sampler[stage], PseudoSampler)
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        
        
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.content_dim=content_dim
        self.text_dim = text_dim
        self.clip_model_path = clip_model_path
        self.max_pseudo_box_num = max_pseudo_box_num
        self.tau = cls_tau
        self.skd_tau = skd_tau
        self.rkd_tau = rkd_tau
        self.alpha = alpha
        self.beta = beta
        self.pre_extracted_clip_text_feat = pre_extracted_clip_text_feat
        self.use_pseudo_box = use_pseudo_box
        self.bg_embedding = nn.Embedding(1, text_dim)
        self.num_additional_padding_prompts = num_additional_padding_prompts
        if novel_obj_queue_dict is not None:
            self.queue = Queues(**novel_obj_queue_dict)
        self.split_visual_text=split_visual_text
        if self.split_visual_text:
            self.visual2text = MLP(text_dim, text_dim*2, text_dim, 3)
            self.fc1 = nn.Linear(text_dim*2, text_dim*2)
            self.fc2 = nn.Linear(text_dim*2, text_dim)
            self.relu = nn.ReLU()
        if use_pseudo_box:
            self.loss_visual_skd = build_loss(loss_visual_skd)
            self.loss_visual_rkd = build_loss(loss_visual_rkd)
        self.use_pre_extracted_clip_text_feat = pre_extracted_clip_text_feat is not None
        self.pre_extracted_clip_text_feat_path = pre_extracted_clip_text_feat
        self.use_text_space_rkd_loss = use_text_space_rkd_loss
        if self.use_text_space_rkd_loss:
            self.class_centroid = None
            self.category_embeddings = None
            self.loss_text_skd = build_loss(loss_visual_skd)
            self.loss_text_rkd = build_loss(loss_visual_rkd)
        self.use_image_level_distill = use_image_level_distill
        if self.use_image_level_distill:
            self.linear_transform = nn.Linear(content_dim, text_dim)
            self.layernorm = nn.LayerNorm(content_dim)
            self.loss_img_distill = build_loss(dict(type='CrossEntropyLoss', loss_weight=0.2))
            self.loss_visual_l1=build_loss(dict(type='L1Loss', loss_weight=5))

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
        
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)
                
        

    def xywh2xyxy(self, boxes):
        """
        boxes: (..., 4) 格式为 (x_center, y_center, width, height)
        返回: (..., 4) 格式为 (x1, y1, x2, y2)
        """
        xy = boxes[..., 0:2]
        wh = boxes[..., 2:4]
        roi = torch.cat([xy - wh * 0.5, xy + wh * 0.5], dim=-1)
        return roi

                
                
    def xyxy2xyzr(self, bbox):
        xy = 0.5 * (bbox[..., 0:2] + bbox[..., 2:4])
        wh = bbox[..., 2:4] - bbox[..., 0:2]
        z = (wh).prod(-1, keepdim=True).sqrt().log2()
        r = (wh[..., 1:2]/wh[..., 0:1]).log2()
        xyzr = torch.cat([xy, z, r], dim=-1)
        return xyzr
                
    def decode_box(self, xyzr):
        # xyzr2xyxy
        scale = 2.00 ** xyzr[..., 2:3]
        ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5,
                                  xyzr[..., 3:4] * 0.5], dim=-1)
        wh = scale * ratio
        xy = xyzr[..., 0:2]
        roi = torch.cat([xy - wh * 0.5, xy + wh * 0.5], dim=-1)
        return roi
    
    def _bbox_forward(self, stage, inter_references, all_bbox_preds, cls_score_features, objnesses):
        # all_bbox_preds归一化 xywh     
        query_xywh = all_bbox_preds[stage]  
        decoded_bboxes = self.xywh2xyxy(query_xywh)
        query_xyzr = self.xyxy2xyzr(decoded_bboxes)
        attn_feats = inter_references[stage]
        # decoded_bboxes = self.decode_box(query_xyzr)
        bboxes_list = [bboxes for bboxes in decoded_bboxes]
        cls_score_feature = cls_score_features[stage]
        cls_score_feature = cls_score_feature.permute(1, 0, 2)
        objness = objnesses[stage]
        objness = objness.permute(1, 0, 2)
        
        bbox_results = dict(
            attn_feats=attn_feats,
            cls_score_feature=cls_score_feature,
            query_xyzr=query_xyzr,
            query_xywh=query_xywh,
            decode_bbox_pred=decoded_bboxes,
            query_content=attn_feats,
            detach_bboxes_list=[item.detach() for item in bboxes_list],
            bboxes_list=bboxes_list,
            objness=objness,
        )
        return bbox_results
    
    def pseudo_logits_from_softmax(self,probs):
        """
        从 softmax 概率反推出伪 logits，支持批量输入
        """
        probs = torch.clamp(probs, min=1e-8)
        log_probs = torch.log(probs)
        mean_log_probs = log_probs.mean(dim=-1, keepdim=True)
        pseudo_logits = log_probs - mean_log_probs  # 去掉平移不变性
        return pseudo_logits
        
    
                
    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            multi_level_feats:
            lastLevel2clip:
        """

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord, \
            multi_level_feats, lastLevel2clip, \
            cls_score_feature, objness= self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        if self.as_two_stage:
            return outputs_classes, outputs_coords, \
                enc_outputs_class, enc_outputs_coord.sigmoid(),\
                multi_level_feats,lastLevel2clip, \
                cls_score_feature, objness, hs
        else:
            return outputs_classes, outputs_coords, \
                None, None,None,None

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             enc_cls_scores,
             enc_bbox_preds,
             multi_level_feats,
             lastLevel2clip,
             cls_score_feature, 
             objness,
             hs,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             all_pseudo_boxes,
             remapping_gt_labels,
             idxs,
             full_category_embeddings=None,
             base_category_embeddings=None,
             gt_bboxes_ignore=None):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        
        # for i, meta in enumerate(img_metas):
        #     print(f"Image {i}:")
        #     for k, v in meta.items():
        #         print(f"  {k}: {v}")
        num_imgs = len(img_metas)
        num_pos = [len(gt_label) for gt_label in gt_labels_list]
        num_pos = torch.tensor(num_pos, dtype=torch.float).sum()
        #num_pos = torch.tensor(num_pos, dtype=torch.float, device=query_content.device).sum()
        num_base_classes = base_category_embeddings.shape[1]
        num_queries= all_cls_scores.shape[2]
        # 全体损失
        loss_dict = dict()
        all_stage_bbox_results = []
        all_stage_assign_results = []
        imgs_whwh = []
        for meta in img_metas:
            img_h, img_w = meta['img_shape'][:2]  # resize 后的大小
            imgs_whwh.append([img_w, img_h, img_w, img_h])  # [w, h, w, h]
        imgs_whwh = torch.tensor(imgs_whwh, dtype=torch.float32, device=base_category_embeddings.device)
        imgs_whwh = imgs_whwh.unsqueeze(1).repeat(1, num_queries, 1) 
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage, hs, all_bbox_preds, cls_score_feature, objness)
            
            all_stage_bbox_results.append(bbox_results)

            query_xyzr = bbox_results['query_xyzr'].detach()
            query_content = bbox_results['query_content']
            bbox_objness = bbox_results['objness']
            query_xywh = bbox_results['query_xywh']
            
            sampling_results = []
            assign_results = []
            with torch.no_grad():
                for i in range(num_imgs):
                    # bbox_objness：类别分数（前景背景）   query_xyzr：query框（xywh）  gt_bboxes_list：真实框   gt_labels_list：类别   img_metas：图   remapping_gt_labels：标签
                    assign_result, sampling_result = self._get_mid_target_single(bbox_objness[i], query_xywh[i], gt_bboxes_list[i], gt_labels_list[i], img_metas[i], remapping_gt_labels, i, gt_bboxes_ignore)
                    sampling_results.append(sampling_result)
                    assign_results.append(assign_result)

            all_stage_assign_results.append(assign_results)

        
        # get pseudo box embedding
        gt_bboxes_with_pseudo_box = []
        gt_labels_with_pseudo_box = []
        all_weighting_score = []
        gt_bboxes_clip_image_feature = []
        all_pseudo_boxes_clip_image_feature = []
        all_pseudo_box_clip_embedding = []
        all_pseudo_proposal_probs = []
        clip_full_image_embedding = []
        clip_full_image_patch_embedding = []
        with torch.no_grad():
            for i in range(num_imgs):
                filename = img_metas[i]['ori_filename']
                filename = filename.split('/')[-1]
                filename = filename[:-4]
                pre_extracted_embedding = torch.load(self.pre_extracted_clip_text_feat_path + filename + '.pth', 'cpu')
                clip_full_image_embedding.append(pre_extracted_embedding['clip_full_image_embedding'])
                clip_full_image_patch_embedding.append(pre_extracted_embedding['clip_full_image_patch_embedding'])
                all_pseudo_proposal_probs.append(pre_extracted_embedding['proposal_probs'])
                if not self.use_pseudo_box:
                    gt_bboxes_with_pseudo_box = gt_bboxes_list
                    gt_labels_with_pseudo_box = remapping_gt_labels
                    all_weighting_score.append(torch.zeros((0,), device=base_category_embeddings.device))
                else:
                    gt_clip_image_feature = pre_extracted_embedding['gt_clip_image_embedding'].to(base_category_embeddings.device).to(base_category_embeddings.dtype)
                    gt_clip_image_feature = F.normalize(gt_clip_image_feature, dim=-1)
                    gt_bboxes_clip_image_feature.append(gt_clip_image_feature)

                    size_mask = ((all_pseudo_boxes[i][:, 2] - all_pseudo_boxes[i][:, 0]) > 32) & ((all_pseudo_boxes[i][:, 3] - all_pseudo_boxes[i][:, 1]) > 32)
                    all_pseudo_boxes[i] = all_pseudo_boxes[i][size_mask][:self.max_pseudo_box_num]

                    pseudo_box_clip_image_feature = pre_extracted_embedding['proposal_clip_image_embedding'].to(base_category_embeddings.device).to(base_category_embeddings.dtype)[size_mask][:self.max_pseudo_box_num]
                    pseudo_box_clip_image_feature = F.normalize(pseudo_box_clip_image_feature, dim=-1)
                    
                    if self.use_pre_extracted_clip_text_feat:
                        pseudo_box_clip_text_feature = pre_extracted_embedding['proposal_clip_text_embedding'].to(base_category_embeddings.device).to(base_category_embeddings.dtype)[size_mask][:self.max_pseudo_box_num]
                        pseudo_box_clip_text_feature = F.normalize(pseudo_box_clip_text_feature, dim=-1)
                    else:
                        pseudo_box_clip_text_feature = pseudo_box_clip_image_feature
                        
                    
                    # weighting_score = torch.sigmoid((torch.diag(pseudo_box_clip_image_feature @ pseudo_box_clip_text_feature.t()) - 0.2) / 0.03) # R50
                    # weighting_score = torch.sigmoid((torch.diag(pseudo_box_clip_image_feature @ pseudo_box_clip_text_feature.t()) - 0.25) / 0.026) # ViT-B-32
                    weighting_score = torch.sigmoid((torch.diag(pseudo_box_clip_image_feature @ pseudo_box_clip_text_feature.t()) - 0.31) / 0.026) * 1.3 # ViT-B-32 detpro
                    #weighting_mask = weighting_score > 0.3
                    #all_pseudo_boxes_clip_image_feature.append(pseudo_box_clip_image_feature[weighting_mask])
                    #all_pseudo_box_clip_embedding.append(pseudo_box_clip_text_feature[weighting_mask])
                    #all_weighting_score.append(weighting_score[weighting_mask])
                    #all_pseudo_boxes[i] = all_pseudo_boxes[i][weighting_mask]
                    all_pseudo_boxes_clip_image_feature.append(pseudo_box_clip_image_feature)
                    all_pseudo_box_clip_embedding.append(pseudo_box_clip_text_feature)
                    all_weighting_score.append(weighting_score)

                    gt_bboxes_with_pseudo_box.append(torch.cat([gt_bboxes_list[i], all_pseudo_boxes[i]]))
                    pseudo_box_label = torch.arange(0, len(all_pseudo_boxes[i]), device=all_pseudo_boxes[i].device, dtype=gt_labels_list[0].dtype) + num_base_classes
                    gt_labels_with_pseudo_box.append(torch.cat([remapping_gt_labels[i], pseudo_box_label]))
            if len(clip_full_image_embedding) > 0:
                clip_full_image_embedding = torch.cat(clip_full_image_embedding, dim=0).to(base_category_embeddings.device).to(base_category_embeddings.dtype)
                clip_full_image_embedding = F.normalize(clip_full_image_embedding, dim=-1)
            if len(clip_full_image_patch_embedding) > 0:
                clip_full_image_patch_embedding = [F.normalize(embed.float().to(base_category_embeddings.device), dim=1).to(base_category_embeddings.dtype) for embed in clip_full_image_patch_embedding]
            clip_full_image_patch_embedding = torch.stack(clip_full_image_patch_embedding, dim=0)
        
        # norms = torch.norm(clip_full_image_patch_embedding, dim=2) 
        # print(norms)
        
        if self.use_pseudo_box and self.use_image_level_distill:
            img_distill_loss = torch.zeros(1, dtype=torch.float, device=query_xyzr.device)
            img_loss_rkd = torch.zeros(1, dtype=torch.float, device=query_xyzr.device)
            img_loss_l1 = torch.zeros(1, dtype=torch.float, device=query_xyzr.device)
            
            encoder_image_features = lastLevel2clip # 传参前已归一化
            saved_encoder_image_query = self.queue.get_queue('encoder_image_query')
            saved_clip_image_query = self.queue.get_queue('clip_image_query')
            skd_logits_1 = encoder_image_features @ torch.cat([clip_full_image_embedding, saved_clip_image_query]).t() * self.skd_tau
            skd_logits_2 = clip_full_image_embedding @ torch.cat([encoder_image_features, saved_encoder_image_query]).t() * self.skd_tau
            img_distill_loss = img_distill_loss + 0.5 * self.loss_img_distill(
                skd_logits_1,
                torch.arange(0, len(clip_full_image_embedding), device=clip_full_image_embedding.device, dtype=torch.long),)
            img_distill_loss = img_distill_loss + 0.5 * self.loss_img_distill(
                skd_logits_2,
                torch.arange(0, len(clip_full_image_embedding), device=clip_full_image_embedding.device, dtype=torch.long),)
            self.queue.dequeue_and_enqueue({'encoder_image_query': encoder_image_features.detach()})
            self.queue.dequeue_and_enqueue({'clip_image_query': clip_full_image_embedding.detach()})
            loss_dict['img_distill_loss'] = img_distill_loss
            sim = torch.sum(encoder_image_features * clip_full_image_embedding, dim=1)  # [4]
            print("对应行相似度:", sim)


            for level in range(4):
                multi_feats = multi_level_feats[level]
                if level == 0:
                    img_loss_l1 = img_loss_l1 + self.loss_visual_l1(multi_feats, clip_full_image_patch_embedding)
                    loss_dict['img_loss_l1'] = img_loss_l1
                
                rkd_logits_multi_feats = torch.bmm(multi_feats, multi_feats.transpose(1,2)) * self.rkd_tau
                rkd_logits_batch_clip = torch.bmm(clip_full_image_patch_embedding, clip_full_image_patch_embedding.transpose(1,2)) * self.rkd_tau
                for i in range(rkd_logits_multi_feats.size(0)):
                    rkd_logits_pred = rkd_logits_multi_feats[0]
                    rkd_logits_clip = rkd_logits_batch_clip[0]
                    img_loss_rkd = img_loss_rkd + self.loss_visual_rkd(
                        rkd_logits_pred,
                        rkd_logits_clip,)
            img_loss_rkd = img_loss_rkd / 4
            loss_dict['img_loss_rkd'] = img_loss_rkd
        

        expanded_class_prompt = []
        if self.use_pseudo_box:
            for i in range(num_imgs):
                pseudo_box_clip_embedding = all_pseudo_box_clip_embedding[i:] + all_pseudo_box_clip_embedding[:i]
                pseudo_box_clip_embedding = torch.cat(pseudo_box_clip_embedding)
                # base_category_embeddings: torch.Size([48, 512])
                # pseudo_box_clip_embedding: torch.Size([8, 512])
                # self.bg_embedding: torch.Size([1, 512])
                expanded_class_prompt.append(torch.cat([base_category_embeddings[i], pseudo_box_clip_embedding, self.bg_embedding(torch.zeros((1,), device=query_content.device, dtype=torch.long))]))
            num_classes = len(pseudo_box_clip_embedding) + num_base_classes
            prompt_len = [len(prompt) for prompt in expanded_class_prompt]
            prompt_len = min(prompt_len)
            expanded_class_prompt = [prompt[:prompt_len] for prompt in expanded_class_prompt]
        else:
            for i in range(num_imgs):
                expanded_class_prompt.append(torch.cat([base_category_embeddings[i], self.bg_embedding(torch.zeros((1,), device=query_content.device, dtype=torch.long)),]))
            num_classes = num_base_classes

        class_prompt = torch.stack(expanded_class_prompt)
        # loss_dict['num_pseudo_box']没指定设备！！！！！！！！！！！！！！！！！！！！！！！！！
        #loss_dict['num_pseudo_box'] = torch.tensor(num_classes - num_base_classes, dtype=torch.float, device=x[0].device) / num_imgs
        loss_dict['num_pseudo_box'] = torch.tensor(num_classes - num_base_classes, dtype=torch.float) / num_imgs

        new_cls_scores = []
        for stage in range(self.num_stages):
            if self.use_text_space_rkd_loss:
                total_unique_label = torch.unique(torch.cat(gt_labels_list))
                if len(total_unique_label) >= 2:
                    base_foreground_query_mask = torch.cat([assign_result.gt_inds > 0 for assign_result in all_stage_assign_results[stage]])
                    gt_num = [len(gt_label) for gt_label in gt_labels_list]
                    gt_num = [0] + gt_num[:-1]
                    # 起始索引偏移量
                    gt_num = torch.cumsum(torch.tensor(gt_num, device=base_foreground_query_mask.device), dim=0)   
                    # 所有前景 query 匹配的 global GT index
                    base_foreground_query_matched_inds = torch.cat([assign_result.gt_inds + gt_num[ii] for ii, assign_result in enumerate(all_stage_assign_results[stage])])[base_foreground_query_mask] - 1
                    total_gt_labels = torch.cat(gt_labels_list, dim=0)
                    total_gt_boxes = torch.cat(gt_bboxes_list, dim=0)
            # image space distillaton loss
            if self.use_pseudo_box:
                bboxes_list = all_stage_bbox_results[stage]['detach_bboxes_list']
                bbox_objness = all_stage_bbox_results[stage]['objness'].clone().detach()
                query_xywh = all_stage_bbox_results[stage]['query_xywh'].clone().detach()
                all_query_embedding = []
                all_clip_embedding = []
                all_novel_query_embedding = []
                novel_proposal_probs = []
                all_novel_proposal_probs = []
                all_novel_proposal_max_probs = []
                for i in range(num_imgs):
                    assign_result = all_stage_assign_results[stage][i]
                    if len(all_pseudo_boxes[i]) > 0:
                        with torch.no_grad():
                            pseudo_box_label = torch.arange(0, len(all_pseudo_boxes[i]), device=all_pseudo_boxes[i].device, dtype=gt_labels_list[0].dtype) + num_base_classes
                            query_inds = torch.arange(0, num_queries, dtype=torch.long, device=all_pseudo_boxes[i].device)
                            query_inds = query_inds[assign_result.gt_inds == 0]
                            # # bbox_objness：类别分数（前景背景）   query_xyzr：query框（xywh）  all_pseudo_boxes：真实框   pseudo_box_label：类别   img_metas：图   remapping_gt_labels：标签
                            # assign_result_second, sampling_result_second = self._get_mid_target_single(bbox_objness[i], query_xywh[i], all_pseudo_boxes[i], pseudo_box_label[i],
                            #                                                                            img_metas[i], remapping_gt_labels, i, gt_bboxes_ignore)

                            bbox_xywh = query_xywh[i][query_inds]
                            assign_result_second = self.assigner.assign(
                                bbox_xywh, bbox_objness[i][query_inds], all_pseudo_boxes[i],
                                pseudo_box_label * 0, img_metas[i])
                            matched_row_inds = assign_result_second.gt_inds > 0
                            matched_col_inds = assign_result_second.gt_inds[matched_row_inds] - 1
                            assign_result_second.labels[matched_row_inds] = pseudo_box_label[matched_col_inds]
                            assign_result_second.gt_inds[matched_row_inds] += len(gt_bboxes_list[i])

                            # merging assignment result
                            assign_result.labels[query_inds] = assign_result_second.labels
                            assign_result.gt_inds[query_inds] = assign_result_second.gt_inds
                            all_stage_assign_results[stage][i] = assign_result
                            # print("gt_inds:", assign_result.gt_inds)     # 每个 query 分配的 gt 索引（0 表示未分配）
                            # print("labels:", assign_result.labels)       # 每个 query 对应的标签（0 是背景或未分配）
                            # print("num_gts:", assign_result.num_gts)     # GT 数量（int）

                            
                    novel_query_embedding = []
                    novel_proposal_probs = []
                    novel_proposal_max_probs = []

                    if torch.sum(assign_result.gt_inds > 0) > 0:
                        query_embedding = all_stage_bbox_results[stage]['cls_score_feature'][i, assign_result.gt_inds > 0]
                        query_embedding = F.normalize(query_embedding, dim=-1)
                        
                        novel_query_mask = assign_result.gt_inds > len(gt_bboxes_list[i])
                        novel_query_embedding.append(all_stage_bbox_results[stage]['cls_score_feature'][i, novel_query_mask])
                        novel_query_embedding = F.normalize(torch.cat(novel_query_embedding), dim=-1)
                    
                        # print("all_pseudo_proposal_probs[i]:",len(all_pseudo_proposal_probs[i]))
                        novel_proposal_probs.append(all_pseudo_proposal_probs[i][assign_result.labels[novel_query_mask] - num_base_classes])
                        novel_proposal_probs = torch.cat(novel_proposal_probs, dim=0)  
                        novel_proposal_max_probs = novel_proposal_probs.max(dim=1).values
                        probs_mask = novel_proposal_max_probs < 0.3

                        
                        clip_image_feature_i = torch.cat([gt_bboxes_clip_image_feature[i], all_pseudo_boxes_clip_image_feature[i]])
                        inds = assign_result.gt_inds.clone()
                        inds[inds > 0] -= 1
                        clip_image_feature_i = clip_image_feature_i[inds]
                        clip_embedding = clip_image_feature_i[assign_result.gt_inds > 0]
      
                        
                        matched_gt_box = gt_bboxes_with_pseudo_box[i][inds][assign_result.gt_inds > 0]
                        matched_pred_box = (bboxes_list[i] * imgs_whwh[i])[assign_result.gt_inds > 0]
                        # print(f"[Image {i}] Matched before IoU filtering: {matched_gt_box.shape[0]}")
                        ious_mask = bbox_overlaps(matched_pred_box, matched_gt_box, is_aligned=True) > 0.5
                        # print(f"[Image {i}] Matched after IoU > 0.5: {ious_mask.sum().item()}")
                        
                        matched_novel_box = gt_bboxes_with_pseudo_box[i][inds][assign_result.gt_inds > len(gt_bboxes_list[i])]
                        matched_novel_pred_box = (bboxes_list[i] * imgs_whwh[i])[assign_result.gt_inds > len(gt_bboxes_list[i])]
                        ious_novel_mask = bbox_overlaps(matched_novel_box, matched_novel_pred_box, is_aligned=True) > 0.5
                        probs_mask = probs_mask.to(ious_novel_mask.device)
                        combined_mask = ious_novel_mask & probs_mask

                        
                        all_query_embedding.append(query_embedding[ious_mask])
                        all_clip_embedding.append(clip_embedding[ious_mask])
                        all_novel_query_embedding.append(novel_query_embedding[combined_mask])
                        # novel_proposal_probs = novel_proposal_probs[combined_mask]
                        all_novel_proposal_probs.append(novel_proposal_probs[combined_mask])
                        # all_novel_proposal_max_probs.append(novel_proposal_max_probs[ious_novel_mask])
                    
                    
                if len(all_clip_embedding) > 0:
                    all_clip_embedding = torch.cat(all_clip_embedding)
                    all_query_embedding = torch.cat(all_query_embedding)
                    # print("all_clip_embedding:",all_clip_embedding.shape)
                if len(all_novel_query_embedding) > 0:
                    all_novel_query_embedding = torch.cat(all_novel_query_embedding)
                    all_novel_proposal_probs = torch.cat(all_novel_proposal_probs)
                    # all_novel_proposal_max_probs = torch.cat(all_novel_proposal_max_probs)
                loss_rkd = torch.zeros(1, dtype=torch.float, device=query_xyzr.device)
                loss_skd = torch.zeros(1, dtype=torch.float, device=query_xyzr.device)
                loss_soft_rkd = torch.zeros(1, dtype=torch.float, device=query_xyzr.device)
                if len(all_clip_embedding) >= 2:
                    saved_obj_query = self.queue.get_queue('obj_query_%d'%(stage))
                    saved_clip_embedding = self.queue.get_queue('clip_query_%d'%(stage))
                    skd_logits_1 = all_query_embedding @ torch.cat([all_clip_embedding, saved_clip_embedding]).t() * self.skd_tau
                    skd_logits_2 = all_clip_embedding @ torch.cat([all_query_embedding, saved_obj_query]).t() * self.skd_tau
                    rkd_logits_clip = all_clip_embedding @ all_clip_embedding.t() * self.rkd_tau
                    rkd_logits_pred = all_query_embedding @ all_query_embedding.t() * self.rkd_tau
                    loss_skd = loss_skd + 0.5 * self.loss_visual_skd(
                        skd_logits_1,
                        torch.arange(0, len(all_clip_embedding), device=all_clip_embedding.device, dtype=torch.long),)
                    loss_skd = loss_skd + 0.5 * self.loss_visual_skd(
                        skd_logits_2,
                        torch.arange(0, len(all_clip_embedding), device=all_clip_embedding.device, dtype=torch.long),)
                    loss_rkd = loss_rkd + self.loss_visual_rkd(
                        rkd_logits_pred,
                        rkd_logits_clip,)
                    # loss_rkd = loss_rkd + self.irm_loss(all_query_embedding, all_clip_embedding) * self.loss_visual_rkd.loss_weight
                    self.queue.dequeue_and_enqueue({'obj_query_%d'%(stage): all_query_embedding.detach()})
                    self.queue.dequeue_and_enqueue({'clip_query_%d'%(stage): all_clip_embedding.detach()})
                
                
                if len(all_novel_query_embedding) > 0:
                    soft_logits_pred = all_novel_query_embedding @ full_category_embeddings.t() * self.tau
                    soft_logits_clip = self.pseudo_logits_from_softmax(all_novel_proposal_probs)
                    # loss_soft_rkd = loss_soft_rkd + self.loss_visual_rkd(
                    #     soft_logits_pred,
                    #     soft_logits_clip,
                    #     all_novel_proposal_max_probs,)
                    loss_soft_rkd = loss_soft_rkd + self.loss_visual_rkd(
                        soft_logits_pred,
                        soft_logits_clip,
                        )

                    

                loss_dict[f'stage{stage}_loss_soft_skd'] = loss_soft_rkd * self.stage_loss_weights[stage]
                loss_dict[f'stage{stage}_loss_skd'] = loss_skd * self.stage_loss_weights[stage]
                loss_dict[f'stage{stage}_loss_rkd'] = loss_rkd * self.stage_loss_weights[stage]
            
            
            

            bbox_results = all_stage_bbox_results[stage]
            cls_score_feature = bbox_results['cls_score_feature']
            if self.split_visual_text:
                t = self.fc1(self.visual2text.layers[-1].weight.clone().detach())
                t_act = self.relu(t)
                transfer_weights = self.fc2(t_act)
                cls_score_feature_rkd = cls_score_feature + F.linear(cls_score_feature, weight=transfer_weights)
                cls_score_feature = cls_score_feature + self.visual2text(cls_score_feature)

            if self.use_text_space_rkd_loss:
                text_space_rkd_loss = torch.zeros(1, dtype=torch.float, device=query_xyzr.device)
                text_space_skd_loss = torch.zeros(1, dtype=torch.float, device=query_xyzr.device)
                if len(total_unique_label) >= 2:
                    query_text_embedding = []
                    # base class query
                    foreground_query = cls_score_feature_rkd.flatten(0, 1)[base_foreground_query_mask]
                    with torch.no_grad():
                        all_pred_bbox = torch.cat(bbox_results['detach_bboxes_list'])[base_foreground_query_mask]
                        ious = bbox_overlaps(all_pred_bbox, total_gt_boxes[base_foreground_query_matched_inds], is_aligned=True).detach()
                    for label_id in total_unique_label:
                        class_mask = total_gt_labels[base_foreground_query_matched_inds] == label_id
                        matched_query = foreground_query[class_mask]
                        class_iou =  torch.softmax(ious[class_mask], dim=0)
                        # class_iou = class_iou.to(matched_query.dtype)
                        fused_class_query = torch.einsum('nh,n->h', matched_query, class_iou)
                        query_text_embedding.append(fused_class_query)
                    query_text_embedding = torch.stack(query_text_embedding, dim=0)
                    normalize_query_text_embedding = F.normalize(query_text_embedding, dim=-1)

                    # novel class query
                    novel_query_embedding = []
                    novel_text_embedding = []
                    novel_query_weight = []

                    for i in range(num_imgs):
                        assign_result = all_stage_assign_results[stage][i]
                        novel_query_mask = assign_result.gt_inds > len(gt_bboxes_list[i])
                        novel_query_embedding.append(cls_score_feature_rkd[i, novel_query_mask])
                        novel_text_embedding.append(all_pseudo_box_clip_embedding[i][assign_result.labels[novel_query_mask] - num_base_classes])
                        novel_query_weight.append(all_weighting_score[i][assign_result.labels[novel_query_mask] - num_base_classes])
                    novel_query_embedding = F.normalize(torch.cat(novel_query_embedding), dim=-1)
                    novel_text_embedding = torch.cat(novel_text_embedding)
                    novel_query_weight = torch.cat(novel_query_weight)

                    padding_labels = torch.unique(torch.cat(idxs))
                    mask = torch.isin(padding_labels, total_unique_label)
                    padding_labels = padding_labels[~mask]
                    saved_centroid_embedding = F.normalize(self.class_centroid[stage].category_embeddings[self.class_centroid[stage].classid_to_idx[padding_labels]], dim=-1)

                    rkd_clip_embedding = torch.cat([self.category_embeddings[total_unique_label], novel_text_embedding, self.category_embeddings[padding_labels]])
                    rkd_query_embedding = torch.cat([normalize_query_text_embedding, novel_query_embedding, saved_centroid_embedding])
                    rkd_logits_clip = rkd_clip_embedding @ rkd_clip_embedding.t() * self.rkd_tau
                    rkd_logits_pred = rkd_query_embedding @ rkd_query_embedding.t() * self.rkd_tau
                    text_space_rkd_loss = text_space_rkd_loss + self.loss_text_rkd(
                        rkd_logits_pred,
                        rkd_logits_clip,)
                    
                    skd_query_embedding = torch.cat([normalize_query_text_embedding, novel_query_embedding])
                    # rkd_clip_embedding = rkd_clip_embedding.to(dtype=skd_query_embedding.dtype)
                    skd_logits1 = skd_query_embedding @ rkd_clip_embedding.t() * self.skd_tau
                    # skd_logits2 = self.base_category_embeddings[total_unique_label] @ rkd_query_embedding.t() * self.skd_tau
                    text_space_skd_loss = text_space_skd_loss + self.loss_text_skd(
                        skd_logits1,
                        torch.arange(0, len(skd_query_embedding), device=total_unique_label.device, dtype=torch.long),
                        torch.cat([torch.ones((len(total_unique_label)), device=total_unique_label.device), novel_query_weight]))
                    # text_space_skd_loss = text_space_skd_loss + 0.5 * self.loss_text_skd(
                    #     skd_logits2,
                    #     torch.arange(0, len(total_unique_label), device=total_unique_label.device, dtype=torch.long),)
                    self.class_centroid[stage].update(total_unique_label, query_text_embedding)
                loss_dict[f'stage{stage}_loss_text_rkd'] = text_space_rkd_loss * self.stage_loss_weights[stage]
                loss_dict[f'stage{stage}_loss_text_skd'] = text_space_skd_loss * self.stage_loss_weights[stage]
            
            cls_score = torch.einsum('nqj,mj->nqm', F.normalize(cls_score_feature, dim=-1), full_category_embeddings) * self.tau
            new_cls_scores.append(cls_score)
        
        
        new_cls_scores = torch.stack(new_cls_scores)
        
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_with_pseudo_box for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_with_pseudo_box for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, new_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)
        
        

        
        
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_with_pseudo_box[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_with_pseudo_box, binary_labels_list,
                                 img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls.to(query_xyzr.device)
            loss_dict['enc_loss_bbox'] = enc_losses_bbox.to(query_xyzr.device)
            loss_dict['enc_loss_iou'] = enc_losses_iou.to(query_xyzr.device)

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1].to(query_xyzr.device)
        loss_dict['loss_bbox'] = losses_bbox[-1].to(query_xyzr.device)
        loss_dict['loss_iou'] = losses_iou[-1].to(query_xyzr.device)
        # loss from other decoder layers
        num_dec_layer = 0

        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i.to(query_xyzr.device)
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i.to(query_xyzr.device)
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i.to(query_xyzr.device)
            num_dec_layer += 1
            
        
        
        return loss_dict

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   enc_cls_scores,
                   enc_bbox_preds,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)
        return result_list

    
    def simple_test(self,
                    mlvl_feats, 
                    img_metas,
                    img_no_normalize,
                    category_embeddings,
                    base_inds_tensor,
                    novel_inds_tensor,
                    rescale=False):
        
        
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        num_imgs = len(img_metas)
        



        
        outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord, multi_level_feats,lastLevel2clip, cls_score_feature, objness, hs = self.forward(mlvl_feats, img_metas)
        
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(
                stage, hs, outputs_coords, cls_score_feature, objness)
            query_content = bbox_results['query_content']
            bboxes_list = bbox_results['detach_bboxes_list']
            
            
        imgs_whwh = []
        for meta in img_metas:
            img_h, img_w = meta['img_shape'][:2]  # resize 后的大小
            imgs_whwh.append([img_w, img_h, img_w, img_h])  # [w, h, w, h]
        imgs_whwh = torch.tensor(imgs_whwh, dtype=torch.float32)
        imgs_whwh = imgs_whwh.unsqueeze(1).repeat(1, 300, 1)
        imgs_whwh = imgs_whwh.to(bboxes_list[0].device)

        cls_score_feature = bbox_results['cls_score_feature']
        bboxes_list = bboxes_list[0] * imgs_whwh[0]
        bboxes = bboxes_list.unsqueeze(0)  # [1, 300, 4]
        bboxes_list = [bboxes]  
        bg_embedding = self.bg_embedding(torch.zeros((1,), device=query_content.device, dtype=torch.long)).unsqueeze(0).repeat(num_imgs, 1, 1)
        category_embeddings = torch.cat([category_embeddings, bg_embedding], dim=1)
        
        if self.split_visual_text:
            cls_score_feature1 = cls_score_feature + self.visual2text(cls_score_feature)
        else:
            cls_score_feature1 = cls_score_feature
        cls_score = torch.einsum('nqj,nmj->nqm', F.normalize(cls_score_feature1, dim=-1), F.normalize(category_embeddings, dim=-1)) * self.tau
        cls_score = cls_score.softmax(-1)
        
        # ensemble
        if self.use_text_space_rkd_loss:
            #print(2)
            t = self.fc1(self.visual2text.layers[-1].weight.clone().detach())
            t_act = self.relu(t)
            transfer_weights = self.fc2(t_act)
            cls_score_feature2 = cls_score_feature + F.linear(cls_score_feature, weight=transfer_weights)
            cls_score2 = torch.einsum('nqj,nmj->nqm', F.normalize(cls_score_feature2, dim=-1), F.normalize(category_embeddings[:, :-1, :], dim=-1)) * self.skd_tau
            cls_score2 = cls_score2.softmax(-1)
        
            cls_score_ens = torch.zeros_like(cls_score)
            cls_score_ens[..., base_inds_tensor] = cls_score[..., base_inds_tensor] ** (1 - self.alpha) * cls_score2[..., base_inds_tensor] ** self.alpha
            cls_score_ens[..., novel_inds_tensor] = cls_score[..., novel_inds_tensor] ** (1 - self.beta) * cls_score2[..., novel_inds_tensor] ** self.beta
            cls_score_ens[..., -1] = cls_score[..., -1] # bg

            # Renormalize the probability to 1.
            cls_score = cls_score_ens / torch.sum(cls_score_ens, dim=-1, keepdim=True)
            
        num_classes = self.num_classes
        det_bboxes = []
        det_labels = []
        
        for img_id in range(num_imgs):
            cls_score_per_img = cls_score[img_id]
            bboxes = bboxes_list[img_id].squeeze(0)  # 原来是 [1,300,4]，用squeeze变 [300,4]
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bboxes /= bboxes.new_tensor(scale_factor)
            det_bbox, det_label, topk_indices = multiclass_nms(bboxes, cls_score_per_img,
                                                        0.005, dict(type='nms', iou_threshold=0.7),
                                                        self.test_cfg.max_per_img, return_inds=True)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        
        


        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(num_imgs)
        ]
        
        return bbox_results