import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class GlobalAlign(nn.Module):
    def __init__(self,
                 in_channel: int = 784,
                 out_channel: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

        self.offset_conv = nn.Conv2d(in_channel, 2, kernel_size=3, stride=1, padding=1)
        
        self.uncertainty_conv = nn.Conv2d(in_channel, 1, kernel_size=3, padding=1)

        self.deform_conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        
        self.softplus = nn.Softplus()
        self.eps = eps

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
       
        img_bev = batch_dict['spatial_features_img'] #[1, 80, 180, 180] (b, c, y, x)
        lidar_bev = batch_dict['spatial_features'] #[1, 256, 180, 180]
        cat_bev = torch.cat([img_bev,lidar_bev],dim=1) #torch.Size([1, 336, 180, 180])
        
        mm_bev = self.conv(cat_bev) #[1, 256, 180, 180]

        if  self.training:#train
            shift_x = random.randint(0, 5)
            shift_y = random.randint(0, 5)
        else:
            shift_x = 0
            shift_y = 9
        shifted_img_bev = torch.roll(img_bev, shifts=(shift_x, shift_y), dims=(3, 2))
        
        feats = torch.cat([shifted_img_bev, lidar_bev], dim=1)
        offset = self.offset_conv(feats) #torch.Size([1, 2, 180, 180])
        sigma_raw = self.uncertainty_conv(feats) 
        # print(offset.shape)
        sigma = self.softplus(sigma_raw) + self.eps


        offset = offset.permute(0, 2, 3, 1) #torch.Size([1, 180, 180, 2])


        deform_weight = F.grid_sample(lidar_bev, offset) #torch.Size([1, 256, 180, 180])
        #print(deform_weight.shape)

        deformed_feature = self.deform_conv(lidar_bev * deform_weight) #([1, 256, 180, 180])
        batch_dict['spatial_features'] = deformed_feature #([1, 256, 180, 180])
        batch_dict['mm_bev_features'] = mm_bev #([1, 256, 180, 180])
        batch_dict['offset_field']        = offset
        batch_dict['uncertainty_field']   = sigma
        return batch_dict
    

    def calculate_loss(self, deformed_feature, mm_bev, sigma):
        #loss_fn = nn.MSELoss()
        diff2 = (deformed - mm_bev).pow(2)
        sigma_sq = sigma.pow(2)
        diff2 = diff2 / (2 * sigma_sq)
        reg  = 0.5 * torch.log(sigma_sq)
        loss = (diff2 + reg).mean()
        #loss = loss_fn(deformed_feature, mm_bev)
        return loss