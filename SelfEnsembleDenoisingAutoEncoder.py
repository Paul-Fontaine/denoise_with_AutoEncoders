import torch
from torchvision import transforms
from AutoEncoderBaseClass import AutoEncoderBaseClass


class SelfEnsembleDenoisingModel(AutoEncoderBaseClass):
    def __init__(self, base_model):
        super(SelfEnsembleDenoisingModel, self).__init__()
        self.base_model = base_model
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90)
        ])

    def forward(self, x):
        outputs = []
        for _ in range(8):  # Apply 8 geometric transformations
            augmented_x = self.transforms(x)
            output = self.base_model(augmented_x)
            outputs.append(output)
        # Average the outputs
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output
