import os
import torch.nn as nn
import torch.optim as optim
from modules.base_networks import *
from modules.HAT_arc import *
# from modules.RCAN_basic import *

import torch
# import model.arch_util as arch_util
import functools

from torchvision.transforms import *
import torch.nn.functional as F


# TDM-short
class TDM_S(nn.Module):

    def __init__(self, nframes, apha=0.5, belta=0.5, nres_b=1):
        super(TDM_S, self).__init__()

        self.nframes = nframes
        self.apha = apha
        self.belta = belta

        base_filter = 128  # bf

        self.feat0 = ConvBlock(3, base_filter, 3, 1, 1, activation='prelu', norm=None)  # 对目标帧特征提取：h*w*3-->h*w*base_filter
        self.feat_diff = ConvBlock(3, 64, 3, 1, 1, activation='prelu', norm=None)  # 对rgb的残差信息进行特征提取：h*w*3 --> h*w*64

        self.conv1 = ConvBlock((self.nframes-1)*64, base_filter, 3, 1, 1, activation='prelu', norm=None)  # 对pooling后堆叠的diff特征增强

        # Res-Block1,h*w*bf-->h*w*64
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(nres_b)]
        modules_body1.append(ConvBlock(base_filter, 64, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)

        # Res-Block1,h*w*bf-->H*W*64，对第一次补充的目标帧特征增强
        modules_body2 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(nres_b)]
        modules_body2.append(ConvBlock(base_filter, 64, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)  # 池化降采样2倍

    def forward(self, lr, neigbor):
        lr_id = self.nframes // 2
        neigbor.insert(lr_id, lr)  # 将中间目标帧插回去
        frame_list = neigbor
        rgb_diff = []
        for i in range(self.nframes-1):
            rgb_diff.append(frame_list[i] - frame_list[i+1])

        rgb_diff = torch.stack(rgb_diff, dim=1)
        B, N, C, H, W = rgb_diff.size()  # [1, nframe-1, 3, 160, 160]

        # 对目标帧及残差图进行特征提取
        lr_f0 = self.feat0(lr)  # h*w*3 --> h*w*256
        diff_f = self.feat_diff(rgb_diff.view(-1, C, H, W))

        down_diff_f = self.avg_diff(diff_f).view(B, N, -1, H//2, W//2)  # 每个diff特征，被降采样2倍[1，4,64,80,80]
        stack_diff = []
        for j in range(N):
            stack_diff.append(down_diff_f[:, j, :, :, :])
        stack_diff = torch.cat(stack_diff, dim=1)
        stack_diff = self.conv1(stack_diff)  # diff 增强

        up_diff1 = self.res_feat1(stack_diff)  # 先过卷积256--》64再上采样

        up_diff1 = F.interpolate(up_diff1, scale_factor=2, mode='bilinear', align_corners=True)  # 使用插值上采样，通道64
        up_diff2 = F.interpolate(stack_diff, scale_factor=2, mode='bilinear', align_corners=True)  # 使用插值上采样，通道还是256

        compen_lr = self.apha * lr_f0 + self.belta * up_diff2

        compen_lr = self.res_feat2(compen_lr)  # 第一次补偿后增强

        compen_lr = self.apha * compen_lr + self.belta * up_diff1

        return compen_lr

# TDM-long
class TDM_L(nn.Module):

    def __init__(self, nframes, apha=0.5, belta=0.5):
        super(TDM_L, self).__init__()

        self.nframes = nframes
        self.apha = apha
        self.belta = belta
        base_filter = 64

        self.compress_3 = ConvBlock(self.nframes*64, base_filter, 3, 1, 1, activation='prelu', norm=None)  # 多尺度压缩3*3：h*w*3-->h*w*256
        # self.compress_5 = ConvBlock(self.nframes*64, base_filter, 5, 1, 2, activation='prelu', norm=None)  # 多尺度压缩5*5：h*w*3-->h*w*256
        # self.compress_7 = ConvBlock(self.nframes*64, base_filter, 7, 1, 3, activation='prelu', norm=None)  # 多尺度压缩7*7：h*w*3-->h*w*256

        self.conv1 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)  # 相减之前的嵌入
        self.conv2 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)  # 相减特征的增强
        self.conv3 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)  # 相减池化特征的增强

        self.conv4 = ConvBlock(base_filter, self.nframes*64, 3, 1, 1, activation='prelu', norm=None)  # 相加之后的增强

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)  # 池化降采样2倍
        self.sigmoid = nn.Sigmoid()


    def forward(self, frame_fea_list):
        frame_fea = torch.cat(frame_fea_list, 1)  # [b nframe*64 h w]

        frame_list_reverse = frame_fea_list
        frame_list_reverse.reverse()  # [[B,64,h,w], ..., ]
        # multi-scale: 3*3
        # forward
        forward_fea3 = self.conv1(self.compress_3(torch.cat(frame_fea_list, 1)))
        # backward
        backward_fea3 = self.conv1(self.compress_3(torch.cat(frame_list_reverse, 1)))
        # 残差
        forward_diff_fea3 = forward_fea3 - backward_fea3
        backward_diff_fea3 = backward_fea3 - forward_fea3

        id_f3 = forward_diff_fea3  # [b 96 h w]
        id_b3 = backward_diff_fea3
        pool_f3 = self.conv3(self.avg_diff(forward_fea3))  # [b 96 h/2, w/2]
        up_f3 = F.interpolate(pool_f3, scale_factor=2, mode='bilinear', align_corners=True)  # 使用插值上采样，通道64

        pool_b3 = self.conv3(self.avg_diff(backward_fea3))
        up_b3 = F.interpolate(pool_b3, scale_factor=2, mode='bilinear', align_corners=True)  # 使用插值上采样，通道64

        enhance_f3 = self.conv2(forward_fea3)
        enhance_b3 = self.conv2(backward_fea3)

        f3 = self.sigmoid(self.conv4(id_f3 + enhance_f3 + up_f3))
        b3 = self.sigmoid(self.conv4(id_b3 + enhance_b3 + up_b3))
        att3 = f3 + b3
        module_fea3 = att3 * frame_fea + frame_fea


        return module_fea3

class TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, nf=64, nframes=5, center=2):
        super(TSA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea

class SwinIR(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, swin_out=64, img_size=50, patch_size=1, in_chans=64,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=4, img_range=1., upsampler='pixelshuffle', resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = swin_out
        num_feat = 64
        self.img_range = img_range

        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection

                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # self.mean = self.mean.type_as(x)
        # x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))

        # x = x / self.img_range + self.mean
        x = x / self.img_range


        return x[:, :, :H * self.upscale, :W * self.upscale]

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class LGTD(nn.Module):
    def __init__(self, nframes):
        super(LGTD, self).__init__()
        # self.swin_out = swin_out  # 投影输出通道数
        self.nframes = nframes
        self.lr_idx = self.nframes // 2
        self.apha = 0.5
        self.belta = 0.5

        self.fea0 = ConvBlock(3, 64, 3, 1, 1, activation='prelu', norm=None)  # 对视频帧特征提取：h*w*3-->h*w*64
        self.fea_all = ConvBlock(3, 64, 3, 1, 1, activation='prelu', norm=None)  # 对视频帧特征提取：h*w*3-->h*w*64
        feature_extraction = [
            ResnetBlock(64, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(5)]
        self.res_feat_ext = nn.Sequential(*feature_extraction)

        self.tdm_s = TDM_S(nframes=self.nframes, apha=self.apha, belta=self.belta)
        self.tdm_l = TDM_L(nframes=self.nframes)
        self.fus = nn.Conv2d(64*self.nframes, 64, 3, 1, 1)
        self.msd = MSD()
        self.TSA_Fusion = TSA_Fusion(64, nframes=self.nframes, center=self.lr_idx)
        # self.swinir = SwinIR(swin_out=self.swin_out)
        self.hat = HAT()
        # self.dconv = DeconvBlock(self.nframes*64, self.swin_out, 8, 4, 2, activation='prelu', norm=None)
        self.embedding = nn.Conv2d(64, 180, 3, 1, 1)
        final_res = [
            ResnetBlock(180, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(5)]
        self.reconstruction = nn.Sequential(*final_res)
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(180, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(4, 64)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # Res-Block2,残差信息增强
        modules_body2 = [
            ResnetBlock(64, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(5)]
        # modules_body2.append(ConvBlock(self.swin_out, self.swin_out, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)
        #
        # # Res-Block3，downsample
        # modules_body3 = [
        #     ResnetBlock(self.swin_out, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
        #     for _ in range(1)]
        # modules_body3.append(ConvBlock(self.swin_out, 64, 8, 4, 2, activation='prelu', norm=None))
        # self.res_feat3 = nn.Sequential(*modules_body3)

        # reconstuction
        # self.conv_reconstuction = nn.Conv2d(self.swin_out*3, 3, 3, 1, 1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, neigbors):
        B, C, H, W = x.size()
        fea_x = self.fea0(x)
        # 首先所有帧输入短期TDM完成补偿
        compen_x = self.tdm_s(x, neigbors)
        # res0 = fea_x - compen_x
        # res0 = self.res_feat2(res0)
        # s_compen_x = fea_x + res0
        # s_compen_x = compen_x + fea_x
        s_compen_x = fea_x + compen_x

        frame_all = neigbors  # TDM_S中已经在neigbor中间插入了目标帧，此时neigbor就是全部帧
        # frame_all.insert(2, x)
        feat_all = torch.stack(frame_all, dim=1)
        feat_all = self.fea_all(feat_all.view(-1, C, H, W))  # 【N 64 ps ps】
        feat_all = self.res_feat_ext(feat_all)
        feat_all = feat_all.view(B, self.nframes, -1, H, W)  # [B, N, 64, ps, ps]

        # 随后MSD配准
        aligned_fea = []
        ref_fea = feat_all[:, self.lr_idx, :, :, :]
        for i in range(self.nframes):
            neigbor_fea = feat_all[:, i, :, :, :]
            aligned_fea.append(self.msd(neigbor_fea, ref_fea))

        # TSA融合
        aligned_fea_all = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        fea = self.TSA_Fusion(aligned_fea_all)  # [b 64 ps ps]

        # 输入长期TDM完成长期多尺度特征补偿
        fram_fea_list = []  # 构建帧特征列表 [b 64 ps ps] * n
        for i in range(self.nframes):
            fram_fea_list.append(aligned_fea[i])  # 这里用配准后的特征可能会好点

        l_compen_x = self.tdm_l(fram_fea_list)  # [b 64*n ps ps]
        l_compen_x = self.fus(l_compen_x)  # [b 64 ps ps]

        # 残差增强
        res = fea - l_compen_x
        res = self.res_feat2(res)
        fea = fea + res

        # res2 = s_compen_x - fea
        # res2 = self.res_feat2(res2)
        # final_compen_x = s_compen_x + res2
        fea = fea + s_compen_x

        # swinblock重建
        # final = self.hat(final_compen_x)
        final = self.embedding(fea)
        final = self.reconstruction(final)
        final = self.conv_before_upsample(final)
        final = self.upsample(final)
        final = self.conv_last(final)

        return final

if __name__ == '__main__':
    input = torch.rand(1, 3, 64, 64).cuda()  # B C H W
    model = LGTD().cuda()
    flops, params = profile(model, inputs=(input,))
    print("Param: {} M".format(params/1e6))
    print("FLOPs: {} G".format(flops/1e9))

    output = model(input)
    print(output.size())