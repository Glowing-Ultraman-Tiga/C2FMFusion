from typing import Any, Dict

import torch
from torch.utils.checkpoint import checkpoint
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel
from .GlobalAlign import GlobalAlign

__all__ = ["BEVFusion"]
        
  
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    @auto_fp16(apply_to=('x',))
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

       
class EfficientAttention(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
    
    #@auto_fp16(apply_to=('x',))
    def forward(self, x):
        B, _, H, W = x.size()
        keys = self.keys(x).reshape(B, self.key_channels, H * W)
        queries = self.queries(x).reshape(B, self.key_channels, H * W)
        values = self.values(x).reshape(B, self.value_channels, H * W)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            k = F.softmax(keys[:, i * head_key_channels:(i + 1) * head_key_channels, :], dim=2)
            q = F.softmax(queries[:, i * head_key_channels:(i + 1) * head_key_channels, :], dim=1)
            v = values[:, i * head_value_channels:(i + 1) * head_value_channels, :]
            context = torch.bmm(k, v.transpose(1, 2))
            attended_value = torch.bmm(context.transpose(1, 2), q)
            attended_value = attended_value.reshape(B, head_value_channels, H, W)
            attended_values.append(attended_value)
        aggregated = torch.cat(attended_values, dim=1)
        reprojected = self.reprojection(aggregated)
        return reprojected + x

class EfficientTransformerFusion(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_heads):
        super().__init__()
        self.proj_in = ConvBNReLU(in_channels, hidden_dim, kernel_size=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        self.efficient_attn = EfficientAttention(hidden_dim, hidden_dim, num_heads, hidden_dim)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        self.proj_out = ConvBNReLU(hidden_dim, hidden_dim, kernel_size=1)
        self.context_fuse = ConvBNReLU(hidden_dim * 2, hidden_dim, kernel_size=1)
        #self.adaptive_fusion = AdaptiveFusion(hidden_dim)

    #@auto_fp16(apply_to=('feat_cam', 'feat_lidar', 'context'))
    def forward(self, feat_cam, feat_lidar, context=None):
        x0 = torch.cat([feat_cam, feat_lidar], dim=1)
        x = self.norm1(checkpoint(self.proj_in, x0))
        if context is None:
            context = torch.zeros_like(x)
        elif context.shape[-2:] != x.shape[-2:]:
            context = checkpoint(F.interpolate, context, x.shape[-2:], 
                               {'mode': 'bilinear', 'align_corners': False})
        x = torch.cat([x, context], dim=1)
        x = checkpoint(self.context_fuse, x)
        #x = checkpoint(self.adaptive_fusion, x)
        x = checkpoint(self.efficient_attn, x) 
        x = self.norm2(checkpoint(self.proj_out, x))
        return x
        


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()
        
        feat_channels = 256
        hidden_dim = 128
        num_heads = 4
        num_layers = 2
        
        #self.global_align = GlobalAlign(in_channel=256, out_channel=256)
        self.transformer_fuse_coarse = EfficientTransformerFusion(256, hidden_dim, num_heads)
        self.transformer_fuse_fine = EfficientTransformerFusion(256, hidden_dim, num_heads)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        #self.w_proj = ConvBNReLU(1, hidden_dim, kernel_size=1)
        
        self.cam_proj_fine = ConvBNReLU(80, 128, kernel_size=1)
        self.lidar_proj_fine = ConvBNReLU(704, 128, kernel_size=1)
        self.cam_proj_coarse = ConvBNReLU(80, 128, kernel_size=1)
        self.lidar_proj_coarse = ConvBNReLU(256, 128, kernel_size=1)

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if encoders.get("radar") is not None:
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

        #if fuser is not None:
        #    self.fuser = build_fuser(fuser)
        #else:
        #    self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']


        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)#[4,6,256,32,88]

        cam_feat = self.encoders["camera"]["vtransform"](
            x,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss, 
            gt_depths=gt_depths,
        )
        return cam_feat
    
    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        sensor_feat = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return sensor_feat
    
    # def extract_lidar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    # def extract_radar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.radar_voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["radar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes
    

    # @torch.no_grad()
    # @force_fp32()
    # def radar_voxelize(self, points):
    #     feats, coords, sizes = [], [], []
    #     for k, res in enumerate(points):
    #         ret = self.encoders["radar"]["voxelize"](res)
    #         if len(ret) == 3:
    #             # hard voxelize
    #             f, c, n = ret
    #         else:
    #             assert len(ret) == 2
    #             f, c = ret
    #             n = None
    #         feats.append(f)
    #         coords.append(F.pad(c, (1, 0), mode="constant", value=k))
    #         if n is not None:
    #             sizes.append(n)

    #     feats = torch.cat(feats, dim=0)
    #     coords = torch.cat(coords, dim=0)
    #     if len(sizes) > 0:
    #         sizes = torch.cat(sizes, dim=0)
    #         if self.voxelize_reduce:
    #             feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
    #                 -1, 1
    #             )
    #             feats = feats.contiguous()

    #     return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        auxiliary_losses = {}
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                cam_feats = self.extract_camera_features(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss:
                    cam_feats, auxiliary_losses['depth'] = cam_feats[0], cam_feats[-1]
            elif sensor == "lidar":
                lidar_feats = self.extract_features(points, sensor)
            elif sensor == "radar":
                radar_feats = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

        if not self.training:
            # avoid OOM
            features = features[::-1]
            
        #proj_cam_coarse = self.cam_proj_coarse(cam_feats[1])
        #proj_lidar_coarse = self.lidar_proj_coarse(lidar_feats[1])
        #batch_dict = {
        #    "spatial_features_img": proj_cam_coarse, 
        #    "spatial_features":    proj_lidar_coarse,
        #}
        #batch_dict = self.global_align(batch_dict)
        #aligned_lidar = batch_dict["spatial_features"]  
        #mm_bev        = batch_dict["mm_bev_features"]
        #sigma = batch_dict['uncertainty_field'] 
          
        
        #loss_align = self.global_align.calculate_loss(aligned_lidar, mm_bev)
        #auxiliary_losses['align'] = loss_align
        
        
        
        #sigma = F.interpolate(
        #    sigma,
        #    size=proj_cam_coarse.shape[-2:],
        #    mode='bilinear',
        #    align_corners=False
        #) 
        #w_proj = checkpoint(self.w_proj, sigma)
       
        with torch.cuda.amp.autocast():
            
            #fused_coarse = self.transformer_fuse_coarse(proj_cam_coarse, aligned_lidar, w_proj)
            fused_coarse = self.transformer_fuse_coarse(self.cam_proj_coarse(cam_feats[1]), self.lidar_proj_coarse(lidar_feats[1]))
        
            fused_coarse_up = self.upsample(fused_coarse)
            fused_fine = self.transformer_fuse_fine(self.cam_proj_fine(cam_feats[0]), self.lidar_proj_fine(lidar_feats[0]), fused_coarse_up)
            x = self.downsample(fused_fine)

        batch_size = x.shape[0]
        #print("BEVshape: ", x.shape)
        
        
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)#解码完变成了高维特征
        
        

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

