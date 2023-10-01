import torch.nn as nn
from typing import Any
from torch.hub import load_state_dict_from_url
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
model_urls = {
    'model': 'https://github.com/dahwin/dahyunfacenet/raw/main/model.pth',
}
__all__ = ['DahwinFaceNet', 'model']
class DahwinFaceNet(nn.Module):
    def __init__(self):
        super(DahwinFaceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=0)
        self.conv9_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=0)
        self.conv10_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv10_2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=0)
        self.conv11_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv11_2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=0)
        self.fc = nn.Linear((64 * 14 * 14), 512//2)  # Removed BatchNorm1d layer
        self.fc1 = nn.Linear(512, 6)

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(F.relu(self.bn6(self.conv6(x))), 2)
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.max_pool2d(x, 2)
        print(x.shape)
        x = x.view(-1, 64 * 14 * 14)
        x = self.fc(x)  # Removed L2 normalization
        return x
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Create the model
# torch.manual_seed(42)
# Model = DahwinFaceNet().to(device)

def model(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DahwinFaceNet:
    r"""
    
    """
    model = DahwinFaceNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['model'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
