import torch
import torch.nn as nn
import torchvision.models as models


class LateFusion(nn.Module):
    def __init__(self, load_pretrained, hyperparameters, fusion_type):
        self.hyperparameters = hyperparameters
        self.fusion_type = fusion_type
        super(LateFusion, self).__init__()
        self.load(load_pretrained)

        # Remove the final 3 classification layers to get spaital features of 4x4x1024 (H'xW'xD)
        if self.fusion_type == "mlp":
            self.base_model = nn.Sequential(*list(self.model.children())[:-1]) # 2048x1x1
        elif self.fusion_type == "avg_pool":
            self.base_model = nn.Sequential(*list(self.model.children())[:-3]) # 1024x4x4

        # Fusion and classification layers
        if self.fusion_type == "mlp":
            # self.fusion_layer = nn.Sequential(
            #     nn.Flatten(),
            #     nn.Linear(2048*1*1*10, 4096),
            #     nn.BatchNorm1d(4096),
            #     nn.ReLU(),
            #     nn.Dropout(0.5),
            #     nn.Linear(4096, 2048),
            #     nn.BatchNorm1d(2048),
            #     nn.ReLU(),
            #     nn.Dropout(0.5),
            #     nn.Linear(2048, 1024),
            #     nn.BatchNorm1d(1024),
            #     nn.ReLU(),
            #     nn.Dropout(0.5),
            #     nn.Linear(1024, self.hyperparameters["num_classes"])
            #     )
            self.fusion_layer = nn.Sequential(
                nn.Flatten(),
                # First reduce spatial dimensions
                nn.Linear(2048*1*1*10, 8192),
                nn.LayerNorm(8192),  # LayerNorm more stable than BatchNorm for large nets
                nn.GELU(),  # GELU often works better than ReLU
                nn.Dropout(0.3),  # Reduced dropout
                
                nn.Linear(8192, 4096),
                nn.LayerNorm(4096),
                nn.GELU(),
                nn.Dropout(0.3),
                
                nn.Linear(4096, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.3),
                
                nn.Linear(1024, self.hyperparameters["num_classes"])
            )

        elif self.fusion_type == "avg_pool":
            self.fusion_layer = nn.Sequential(
                nn.AdaptiveAvgPool3d(
                    (1, 1, 1)
                ),  # Will reduce T, H', W' to 1 so now we have BxDx1x1x1 -> 8x1024x1x1x1
                nn.Flatten(),  # 8x1024
                nn.Linear(1024, self.hyperparameters["num_classes"]),
            )
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

    def load(self, load_pretrained):
        if load_pretrained:
            self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            self.model = models.resnet152(weights=None)

    def forward(self, x):
        # x shape: [batch_size, num_channels, num_frames, height, width]
        batch_size, channels, num_frames, height, width = x.shape

        # Extract features for each frame
        features = []
        for frame_idx in range(num_frames):
            frame = x[:, :, frame_idx, :, :]
            frame_features = self.base_model(frame)
            features.append(frame_features)

        # Stack features
        features = torch.stack(features, dim=1)  # BxTxDxH'xW' -> 8x10x1024x4x4

        # Fusion based on type
        if self.fusion_type == "mlp":
            # Flatten across frames and features
            x = features.reshape(batch_size, -1)  # Flatten Bx(TxDxH'xW')
            output = self.fusion_layer(x)
        elif self.fusion_type == "avg_pool":
            x = features.permute(0, 2, 1, 3, 4)  # Now BxDxTxH'xW' -> 8x1024x10x4x4
            # Perform average pooling over space (H' and W') + time (T) with D (1024) reamining fature dimension
            output = self.fusion_layer(x)
        return output


############################# TESTING #############################
# if __name__ == '__main__':
#     # Hyperparameters
#     hp = {
#         'num_classes': 10,
#     }

#     # Create models with different fusion types
#     mlp_fusion_model = LateFusion(load_pretrained=True, fusion_type='mlp', hyperparameters=hp)
#     avg_pool_fusion_model = LateFusion(load_pretrained=True, fusion_type='avg_pool', hyperparameters=hp)

#     # Input (batch_size, channels, frames, height, width)
#     x = torch.randn(8, 3, 10, 64, 64)

#     # Test MLP fusion
#     mlp_output = mlp_fusion_model(x)
#     print("MLP Fusion Output Shape:", mlp_output.shape)

#     # Test Average Pooling fusion
#     avg_pool_output = avg_pool_fusion_model(x)
#     print("Avg Pool Fusion Output Shape:", avg_pool_output.shape)
