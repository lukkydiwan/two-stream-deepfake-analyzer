
class SRMLayer(nn.Module):
    def __init__(self):
        super(SRMLayer, self).__init__()
        # 3 standard SRM filters to capture noise residuals
        srm_kernel = np.array([
            [[-1,  2, -1], [ 2, -4,  2], [-1,  2, -1]], 
            [[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]], 
            [[ 0, -1,  0], [-1,  4, -1], [ 0, -1,  0]]  
        ], dtype=np.float32) / 4.0
        
        self.kernel = torch.tensor(srm_kernel).view(3, 1, 3, 3).repeat(1, 3, 1, 1)
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1)

class TwoStreamNet(nn.Module):
    def __init__(self):
        super(TwoStreamNet, self).__init__()
        # Stream 1: Spatial (Standard RGB)
        self.spatial_net = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Stream 2: Noise (SRM)
        self.srm_layer = SRMLayer()
        self.noise_net = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Fusion Classifier
        self.fc = nn.Sequential(
            nn.Linear(1280 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # RGB Stream Feature Extraction
        feat1 = self.spatial_net.extract_features(x)
        feat1 = F.adaptive_avg_pool2d(feat1, 1).view(feat1.size(0), -1)
        
        # Noise Stream Feature Extraction (On-the-fly SRM)
        noise = self.srm_layer(x)
        feat2 = self.noise_net.extract_features(noise)
        feat2 = F.adaptive_avg_pool2d(feat2, 1).view(feat2.size(0), -1)
        
        # Concatenation and Final Prediction
        combined = torch.cat((feat1, feat2), dim=1)
        return self.fc(combined)

# Initialize the model
model = TwoStreamNet().to(device)