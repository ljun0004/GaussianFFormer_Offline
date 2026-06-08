import torch
import torch.nn.functional as F
from functools import partial
from mmseg.models import SEGMENTORS
from .base_segmentor import CustomBaseSegmentor

from ..encoder.gaussian_encoder.utils import cartesian
from ..utils.safe_ops import safe_sigmoid

class GaussianContainer:
    """
    A structured namespace container that enforces physical mathematical bounds 
    on the free parameters before they enter the differentiable renderer.
    This acts as the direct equivalent of the 'GaussianPrediction' object.
    """
    def __init__(
        self, 
        x_bar, 
        pc_range, 
        scale_range,
        include_opa=True
    ):
        # 1. Means (XYZ) [Indices 0:3]:
        # Native 1:1 match with 'xyz = self.get_xyz(anchor_xyz)'
        get_xyz_func = partial(cartesian, pc_range=pc_range, use_sigmoid=True)
        x_world = get_xyz_func(x_bar[..., 0:3])
        
        # Keep the CUDA boundary clamps to prevent local_aggregate_prob assertion faults
        eps = 1e-3
        x_clamp = torch.clamp(x_world[..., 0:1], pc_range[0] + eps, pc_range[3] - eps)
        y_clamp = torch.clamp(x_world[..., 1:2], pc_range[1] + eps, pc_range[4] - eps)
        z_clamp = torch.clamp(x_world[..., 2:3], pc_range[2] + eps, pc_range[5] - eps)
        self.means = torch.cat([x_clamp, y_clamp, z_clamp], dim=-1)
        
        # 2. Scales [Indices 3:6]:
        # Native 1:1 match with original scale un-normalization
        raw_scales = x_bar[..., 3:6]
        self.scales = scale_range[0] + (scale_range[1] - scale_range[0]) * safe_sigmoid(raw_scales)
        
        # 3. Rotations [Indices 6:10]:
        # Native 1:1 match with original L2 normalization
        raw_rots = x_bar[..., 6:10]
        self.rotations = F.normalize(raw_rots, p=2, dim=-1)
        
        # 4. Opacity [Indices 10:11] (if enabled):
        # Native 1:1 match with 'opacities=safe_sigmoid(anchor_opa)'
        semantic_start = 10 + int(include_opa)
        if include_opa:
            raw_opacities = x_bar[..., 10:11]
            self.opacities = safe_sigmoid(raw_opacities)
        else:
            self.opacities = torch.ones_like(x_bar[..., 10:11])
        
        # 5. Semantics [Indices 11 or 10:]:
        # Native 1:1 match with semantics_activation='identity'
        self.semantics = x_bar[..., semantic_start:]


@SEGMENTORS.register_module()
class InverseGraphicsSegmentor(CustomBaseSegmentor):
    def __init__(
        self, 
        pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0], 
        scale_range=[0.01, 1.8],
        num_gaussians=25600,
        semantic_dim=17,
        include_opa=True,
        **kwargs
    ):
        # Explicitly strip out the transformer encoder to prevent VRAM allocation
        kwargs.pop('encoder', None)
        super().__init__(**kwargs)
        
        # Save hyper-dimensions mapped from the config file as class properties
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.num_gaussians = num_gaussians
        self.semantic_dim = semantic_dim
        self.include_opa = include_opa
        
        # Lock down gradient tracking on the vision components used for initialization
        if hasattr(self, 'img_backbone') and self.img_backbone is not None:
            self.img_backbone.requires_grad_(False)
            self.img_backbone.eval()
        if hasattr(self, 'img_neck') and self.img_neck is not None:
            self.img_neck.requires_grad_(False)
            self.img_neck.eval()
        if hasattr(self, 'lifter') and self.lifter is not None:
            self.lifter.requires_grad_(False)
            self.lifter.eval()

    def extract_img_feat(self, imgs, **kwargs):
        """Extract multi-scale image features matching the BEVSegmentor flow."""
        B = imgs.size(0)
        result = {}

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        
        img_feats = []
        # Pull specified out_indices from configuration backbone properties
        out_indices = getattr(self, 'img_backbone_out_indices', [0, 1, 2, 3])
        for idx in out_indices:
            img_feats.append(img_feats_backbone[idx])
            
        img_feats = self.img_neck(img_feats)
        if isinstance(img_feats, dict):
            secondfpn_out = img_feats["secondfpn_out"][0]
            BN, C, H, W = secondfpn_out.shape
            secondfpn_out = secondfpn_out.view(B, int(BN / B), C, H, W)
            img_feats = img_feats["fpn_out"]
            result.update({"secondfpn_out": secondfpn_out})

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        result.update({'ms_img_feats': img_feats_reshaped})
        return result

    def get_cold_start(self, imgs=None, metas=None, **kwargs):
        """
        Executes a purely mathematical initialization, completely bypassing 
        the vision backbone and lifter. 
        Returns an optimizer-ready nn.Parameter for the current scene.
        """
        if imgs is not None:
            B = imgs.size(0)
            device = imgs.device
        else:
            B = 1
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            # 1. Means (XYZ Logits) dynamically generated via config properties
            xyz_logits = (torch.rand(B, self.num_gaussians, 3, device=device) - 0.5) * 4.0
            scale_logits = torch.zeros(B, self.num_gaussians, 3, device=device)
            
            rotations = torch.zeros(B, self.num_gaussians, 4, device=device)
            rotations[..., 0] = 1.0
            
            # Dynamic opacity inclusion
            if self.include_opa:
                opacity_logits = torch.zeros(B, self.num_gaussians, 1, device=device)
            else:
                opacity_logits = torch.empty((B, self.num_gaussians, 0), device=device)
                
            semantic_logits = torch.randn(B, self.num_gaussians, self.semantic_dim, device=device)
            
            x_bar_init = torch.cat([
                xyz_logits, scale_logits, rotations, opacity_logits, semantic_logits
            ], dim=-1)
            
        return torch.nn.Parameter(x_bar_init, requires_grad=True)

    def get_warm_start(self, imgs, metas, **kwargs):
        """
        Executes the pre-trained distribution-based initialization.
        Returns an optimizer-ready nn.Parameter for the current scene.
        """
        with torch.no_grad():
            results = {'imgs': imgs, 'metas': metas}
            # outs = self.extract_img_feat(imgs=imgs)
            # results.update(outs)
            
            outs = self.lifter(**results)
            x_bar_init = outs['representation'].clone().detach()
            
        return torch.nn.Parameter(x_bar_init, requires_grad=True)

    def forward(self, x_bar, metas, **kwargs):
        """
        Routes the explicit scene parameter tensor straight to the custom CUDA 
        probabilistic aggregation kernel, completely bypassing the transformer encoder.
        """
        # Pass the dynamically mapped config properties into the container
        mock_gaussian = GaussianContainer(
            x_bar, 
            pc_range=self.pc_range,
            scale_range=self.scale_range,
            include_opa=self.include_opa
        )
        
        representation = [{'gaussian': mock_gaussian}]
        outs = self.head(representation=representation, metas=metas)
        return outs