import math
import warnings
from functools import partial

import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.nn.init import normal_
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import constant_, xavier_uniform_
from .ctc import CrossAttention
import MultiScaleDeformableAttention as MSDA

from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd
import torch.nn.functional as F


def cub_cvit(backbone_name="vit_large_patch16_224", baseline=False, *args, **kwargs):
    """
    Args:
        baseline (bool): If true it returns the baseline model, which in this case it just the vit backbone without concept transformer
    """
    if not baseline:
        return CVIT(
            model_name=backbone_name,
            num_classes=200,
            n_unsup_concepts=0,
            n_concepts=13,
            n_spatial_concepts=95,
            num_heads=12,
            attention_dropout=0.1,
            projection_dropout=0.1,
            *args,
            **kwargs,
        )
    else:
        return ExplVIT(model_name=backbone_name, num_classes=200)


class ExplVIT(VisionTransformer):
    """VIT modified to return dummy concept attentions"""
    def __init__(self, num_classes=200, model_name="vit_base_patch16_224", *args, **kwargs):
        super().__init__(num_classes=num_classes)

        loaded_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.load_state_dict(loaded_model.state_dict())

    def forward(self, x):
        out = super().forward(x)
        return out, None, None, None


class CVIT(nn.Module):
    """Concept Vision Transformer"""
    def __init__(
        self,
        num_classes=200,
        model_name="vit_base_patch16_224",
        pretrained=True,
        n_unsup_concepts=50,
        n_concepts=13,
        n_spatial_concepts=95,
        num_heads=12,
        attention_dropout=0.1,
        projection_dropout=0.1,
        psi = 1,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.feature_extractor = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        del self.feature_extractor.head

        self.classifier = ConceptTransformerVIT(
            embedding_dim=self.feature_extractor.embed_dim,
            num_classes=num_classes,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            n_unsup_concepts=n_unsup_concepts,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

        embed_dim = 1024
        self.psi = psi
        print("Psi:", psi)
        self.spm = SpatialPriorModule(inplanes=64, embed_dim=embed_dim)
        self.spm.apply(self._init_weights)
        self.map_spatial_vit = nn.Sequential(
                        nn.Conv2d(in_channels = 1029,out_channels = 512, kernel_size = 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels = 512,out_channels = 197, kernel_size = 1)
                        )
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        normal_(self.level_embed)

        # self.deform_attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
        #                          n_points=n_points, ratio=deform_ratio)

        self.injector = Injector(dim=1024, n_levels=3, num_heads=8, init_values=0.,
                                 n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                 deform_ratio=1.0,
                                 with_cp=False)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        # Spatiality calculations
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        spatial_feat = torch.cat([c2, c3, c4], dim=1)

        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)

        # if self.feature_extractor.dist_token is None:
        #     x = torch.cat((cls_token, x), dim=1)
        # else:
        #     x = torch.cat((cls_token, self.feature_extractor.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        x = self.feature_extractor.blocks(x)
        x = self.feature_extractor.norm(x)



        ### Add spm output - naive
        # x = x + self.psi * self.map_spatial_vit(spatial_feat.unsqueeze(-1)).squeeze(-1)

        ### Add spm output - attention
        spatials = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=spatial_feat, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        


        ### Add spm output - deform-attention
        x = x + self.psi * spatials



        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.classifier(
            x[:, 0].unsqueeze(1), x[:, 1:]
        )
        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class ConceptTransformerVIT(nn.Module):
    """Processes spatial and non-spatial concepts in parallel and aggregates the log-probabilities at the end.
    The difference with the version in ctc.py is that instead of using sequence pooling for global concepts it
    uses the embedding of the cls token of the VIT
    """
    def __init__(
        self,
        embedding_dim=768,
        num_classes=10,
        num_heads=2,
        attention_dropout=0.1,
        projection_dropout=0.1,
        n_unsup_concepts=10,
        n_concepts=10,
        n_spatial_concepts=10,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Unsupervised concepts
        self.n_unsup_concepts = n_unsup_concepts
        self.unsup_concepts = nn.Parameter(
            torch.zeros(1, n_unsup_concepts, embedding_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.unsup_concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_unsup_concepts > 0:
            self.unsup_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Non-spatial concepts
        self.n_concepts = n_concepts
        self.concepts = nn.Parameter(torch.zeros(1, n_concepts, embedding_dim), requires_grad=True)
        nn.init.trunc_normal_(self.concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_concepts > 0:
            self.concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Spatial Concepts
        self.n_spatial_concepts = n_spatial_concepts
        self.spatial_concepts = nn.Parameter(
            torch.zeros(1, n_spatial_concepts, embedding_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.spatial_concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_spatial_concepts > 0:
            self.spatial_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def forward(self, x_cls, x):
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        out = 0.0
        if self.n_unsup_concepts > 0:  # unsupervised stream
            out_unsup, unsup_concept_attn = self.concept_tranformer(x_cls, self.unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.squeeze(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream
            out_n, concept_attn = self.concept_tranformer(x_cls, self.concepts)
            concept_attn = concept_attn.mean(1)  # average over heads
            out = out + out_n.squeeze(1)  # squeeze token dimension

        if self.n_spatial_concepts > 0:  # Spatial stream
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, self.spatial_concepts)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
            out = out + out_s.mean(1)  # pool tokens sequence

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = nn.ReLU(inplace=True)(self.fc1(c1))
        c2 = nn.ReLU(inplace=True)(self.fc2(c2))
        c3 = nn.ReLU(inplace=True)(self.fc3(c3))
        c4 = nn.ReLU(inplace=True)(self.fc4(c4))

        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4
    


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttnFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(value, value_spatial_shapes,
                                             value_level_start_index,
                                             sampling_locations,
                                             attention_weights,
                                             ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, \
        sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        """Multi-Scale Deformable Attention Module.

        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, '
                             'but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2
        # which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make "
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] *
                input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        value = value.view(N, Len_in, self.n_heads,
                           int(self.ratio * self.d_model) // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).\
            view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'
                .format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index,
                                            sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output
    

class Injector(nn.Module):
    def __init__(self, dim, num_heads=16, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.01, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)

        # self.final_norm = norm_layer(dim)

        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        # breakpoint()
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):

        # Seperate the last cls token (TODO: do less hard coding) 
        cls_seperate = query[:,196:,:]
        query = query[:,:196,:]

        def _inner_forward(query, feat):
            
            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn
        
        # if self.with_cp and query.requires_grad:
        #     query = cp.checkpoint(_inner_forward, query, feat)
        # else:
        query = _inner_forward(query, feat)
        # query = self.final_norm(query)
        # Reattach the cls token
        query = torch.cat((query,cls_seperate),dim=1)
        
        return query


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                                   (h // 16, w // 16),
                                                   (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2

