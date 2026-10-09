import torch
import torch.nn as nn
import torchvision.models as models

class MedicalImageClassifier(nn.Module):
    """
    Unified model class
    """
    def __init__(self, model_name: str = 'resnet', pretrained: bool = True, num_classes: int = 3):
        super(MedicalImageClassifier, self).__init__()
        self.model_name = model_name.lower()
        self.pretrained = pretrained
        self.num_classes = num_classes

        self.model = self._build_model()

    def _build_model(self):
        # 1. AlexNet
        if self.model_name == 'alexnet':
            weights = models.AlexNet_Weights.DEFAULT if self.pretrained else None
            model = models.alexnet(weights=weights)
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, self.num_classes)

        # 2. ResNet
        elif self.model_name == 'resnet':
            weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)

        # 3. GoogLeNet
        elif self.model_name == 'googlenet':
            weights = models.GoogLeNet_Weights.DEFAULT if self.pretrained else None
            model = models.googlenet(weights=weights, aux_logits=True)

            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)

            model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, self.num_classes)
            model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}. Choose one of 'alexnet', 'resnet', 'googlenet'")

        return model

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, outputs, labels, criterion):
        """
        Model specific loss computation 
        """
        if self.model_name == 'googlenet' and self.training:
            # GoogLeNetOutputs or tuple: (main, aux2, aux1)
            main_out, aux2_out, aux1_out = outputs
            loss_main = criterion(main_out, labels)
            loss_aux2 = criterion(aux2_out, labels)
            loss_aux1 = criterion(aux1_out, labels)
            
            return loss_main + 0.3 * loss_aux2 + 0.3 * loss_aux1
            
        return criterion(outputs, labels)

    def get_predictions(self, outputs):
        """
        Logit extraction to compute final accuracy
        """
        if self.model_name == 'googlenet' and self.training:
            return outputs[0]
            
        return outputs

if __name__ == '__main__':
    resnet_pretrained = MedicalImageClassifier(model_name='resnet', pretrained=True, num_classes=3)
    print("ResNet (Pretrained) loaded")

    googlenet_scratch = MedicalImageClassifier(model_name='googlenet', pretrained=True, num_classes=3)
    print("GoogLeNet (Pretrained) loaded")

    alexnet_pretrained = MedicalImageClassifier(model_name='alexnet', pretrained=True, num_classes=3)
    print("AlexNet (Pretrained) loaded")

    # Tensor input test (Batch Size 8, 3 Channels, 299x299 Image)
    dummy_input = torch.randn(8, 3, 299, 299)
    output = resnet_pretrained(dummy_input)
    print(f"Output tensor shape: {output.shape}")
