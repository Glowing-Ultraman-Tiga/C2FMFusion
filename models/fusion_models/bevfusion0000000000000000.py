from typing import Any, Dict

import torch
import numpy as np
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
from mmdet3d.Match.match import FeatureMatchingModule
from scipy.spatial.transform import Rotation as R

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
from nuscenes import NuScenes

__all__ = ["BEVFusion"]

# 设置数据集路径
data_root = "/data/yali_data/bevfusion-main/data/nuscenes/"
nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)


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

        self.encoders = nn.ModuleDict()
        self.feature_matching = FeatureMatchingModule()
        self.previous_bev_feature = None
        self.previous_t = None
        self.previous_R = None

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

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

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
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in [
            'BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def get_pose(self, nusc, sample_token):
        """
        获取给定sample_data_token的旋转和平移信息
        """
        # print("Loaded tables:", nusc.table_names)  # 检查是否加载了完整的表

        # if sample_token not in nusc._token2ind['sample_data']:
        # raise ValueError(f"Token '{sample_token}' not found in the NuScenes database.")

        # 获取sample_data对应的位姿token
        # sd_record = nusc.get('sample_data', sample_token)
        # matched_data = next((item for item in sample_data_list if item.get("sample_token") == target_sample_token), None)
        sd_record = None
        for record in nusc.sample_data:
            if record["sample_token"] == sample_token:
                sd_record = record
        # print("1111111111111", sd_record)
        pose_token = sd_record['ego_pose_token']

        # 获取位姿信息
        pose_record = nusc.get('ego_pose', pose_token)

        # 提取旋转和位移
        translation = pose_record['translation']  # [x, y, z]
        rotation = pose_record['rotation']  # [w, x, y, z]
        #print("000000004translation", translation)
        #print("000000003rotation", rotation)

        return translation, rotation

    def quaternion_to_rotation_matrix(q):
        """
        将四元数转换为旋转矩阵
        """
        import numpy as np

        w, x, y, z = q
        R = np.array([
            [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
        ])
        return R

    def update_pose(self, prev_R, prev_t, R_rel, t_rel):
        """
        根据上一帧的位姿和相对位姿更新当前帧的全局位姿
        """
        # 将 prev_R 四元数转换为旋转矩阵
        prev_R_matrix = R.from_quat(prev_R).as_matrix()

        # 将相对旋转四元数 R_rel 转换为旋转矩阵
        #R_rel_matrix = R.from_quat(R_rel).as_matrix()
        R_rel_matrix = R_rel
        t_rel = np.array(t_rel)  # 确保 t_rel 是一维数组
    
        #print("R_rel_matrix:", R_rel_matrix)  # 检查 R_rel 的值和形状
        #print("t_rel:", t_rel)  # 检查 t_rel 的值和形状

        # 计算预测的全局旋转矩阵并转换为四元数
        R_pred_matrix = np.dot(prev_R_matrix, R_rel_matrix)
        R_pred = R.from_matrix(R_pred_matrix).as_quat()  # 转换为四元数

        # 计算预测的全局平移向量
        t_pred = prev_t + np.dot(prev_R_matrix, t_rel)

        return R_pred, t_pred

    def compute_pose_loss(self, R_pred, t_pred, R_true, t_true):
        """
        计算位姿损失，包括旋转误差和平移误差
        """
        #print("111111111", R_true)
        #print("111111112", t_true)
        # R_true = np.array(R_true)
        # R_pred = np.array(R_pred)
        R_true = R.from_quat(R_true).as_matrix()
        R_pred = R.from_quat(R_pred).as_matrix()

        # 计算旋转误差（角度差）
        R_error = np.dot(R_true.T, R_pred)
        trace = np.trace(R_error)
        angle_error = np.arccos((trace - 1) / 2)  # 计算旋转矩阵之间的角度差

        # 计算平移误差（欧氏距离）
        t_error = np.linalg.norm(t_pred - t_true)

        # 将旋转误差和平移误差组合为总损失
        pose_loss = angle_error + t_error
        
        #print("88888888888888888888", pose_loss)

        return pose_loss

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
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
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
        return x

    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

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
            outputs, x = self.forward_single(
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
            
            #original_loss = outputs.get("loss", 0)
            
            #batch_size = x.shape[0]
            if self.training:
                #if isinstance(outputs, list):
                 #   outputs = {"loss": sum(item.get("loss", 0) for item in outputs if isinstance(item, dict))}
                #else:
                 #   original_loss = outputs.get("loss", 0)
                original_loss = outputs.get("loss", 0) if isinstance(outputs, dict) else sum(item.get("loss", 0) for item in outputs if isinstance(item, dict))
                if self.previous_bev_feature is None:
                    #print("000000000000000", metas)
                    sample_token = metas[0]["token"]
                    self.previous_t, self.previous_R = self.get_pose(nusc, sample_token)
                    return outputs
                # Perform feature matching if previous BEV feature exists
                else:
                    R_rel, t_rel, matches = self.feature_matching.match(self.previous_bev_feature, x)
                    # 在调用update_pose之前检查R_rel和t_rel
                    if R_rel is not None and t_rel is not None:
                        R_pred, t_pred = self.update_pose(self.previous_R, self.previous_t, R_rel, t_rel)
                    else:
                        # 如果R_rel或t_rel为None，输出提示信息并跳过本次位姿更新
                        print("特征匹配失败，跳过本次位姿更新")
                        R_pred, t_pred = self.previous_R, self.previous_t
                        
                        # print("2222222222", metas)
                    sample_token = metas[0]["token"]
                    # print(f"Sample token extracted: {sample_token}")
    
                    t_true, R_true = self.get_pose(nusc, sample_token)
    
                    # Compute pose loss
                    pose_loss = self.compute_pose_loss(R_pred, t_pred, R_true, t_true)
                    #print("outputs:", outputs.type, outputs)
                    outputs["loss/pose"] = torch.tensor(pose_loss).float().cuda()
                    total_loss = original_loss + cfg.loss_weights.pose_loss * pose_loss
                    # else:
                    # total_loss = original_loss
    
                    # Save current feature for next step
                    self.previous_bev_feature = x
                    # self.previous_R, self.previous_t = R_pred, t_pred
                    #losses = losses + total_loss
    
                    return {"loss": total_loss}, outputs
            else:
                #print("11111111111111", metas)
                sample_token = metas[0]["token"]
                self.previous_t, self.previous_R = self.get_pose(nusc, sample_token) if self.previous_bev_feature is None else self.previous_bev_feature
                outputs = {
                    # ...（其他输出，例如检测结果）...
                    "predicted_pose": {
                        "rotation": R_pred.tolist(),
                        "translation": t_pred.tolist()
                    },
                    "true_pose": {
                        "rotation": R_true.tolist(),
                        "translation": t_true.tolist()
                    },
                    # 可选：如果您计算了旋转和平移误差，也可以包含在输出中
                    "pose_loss": pose_loss,
                }
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
                feature = self.extract_camera_features(
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
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]
        print("BEVshape:", x.shape)
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)
        print("xshape:", x)
        

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
            #print("11111111111111133", outputs.type)
            return outputs, x
        else:
            #outputs = [{} for _ in range(batch_size)]
            #outputs = {}
            outputs = {k: {} for k in range(batch_size)}
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
                    
                

                #self.previous_bev_feature = x  # 更新上一帧BEV特征
            #print("11111111111111122", outputs.type)
            return outputs, x

