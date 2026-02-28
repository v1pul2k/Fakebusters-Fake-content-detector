import torch
import torch.nn as nn
import timm

class XceptionAvgTemporal(nn.Module):
    def __init__(self, backbone_name="xception41", pretrained=True):
        super().__init__()

        # Load Xception backbone (no classifier)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''    # no built-in pooling
        )

        # Feature dim determined from backbone
        self.feat_dim = self.backbone.num_features

        # Our pooling layer
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification head (raw logits)
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def extract_features(self, x):
        """
        Safely extract features regardless of timm Xception version.
        """

        # Preferred method
        if hasattr(self.backbone, "forward_features") and callable(getattr(self.backbone, "forward_features")):
            return self.backbone.forward_features(x)

        # Fallback for many Xception variants
        if hasattr(self.backbone, "features") and callable(getattr(self.backbone, "features")):
            return self.backbone.features(x)

        # Final fallback - full forward
        return self.backbone(x)

    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # Merge time dimension
        x = x.reshape(B * T, C, H, W)

        # Extract CNN features
        feats = self.extract_features(x)

        # Spatial pooling
        feats = self.pool(feats)            # [B*T, C, 1, 1]
        feats = feats.view(B, T, self.feat_dim)

        # Temporal averaging
        feats = feats.mean(dim=1)

        # Return logits (for BCEWithLogitsLoss)
        return self.head(feats)
