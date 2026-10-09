import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from dataset import load_data, get_loaders
from models import MedicalImageClassifier
from train import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="AI4MIA Week5: COVID-19 Image Classification Project")
    
    # Directory configurations
    parser.add_argument('--raw_dir', type=str, default='dataset/raw', help='Path to raw dataset directory')
    parser.add_argument('--processed_dir', type=str, default='dataset/processed', help='Path to preprocessed dataset directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Path to root directory of resultant directories')
    
    # Data loading configurations
    parser.add_argument('--use_saved', action='store_true', help='Use preprocessed .pt tensor dataset')
    parser.add_argument('--save_processed', action='store_true', help='Save raw data to .pt tensors')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader parameter')
    parser.add_argument('--augment', type=str, default='none',
                        choices=['none', 'all', 'flip', 'rotation', 'crop'], help="Data augmentation to train with more data. 'all' means all possible augmentations. Default is 'none'")
    
    # Model selection and hyperparameters
    parser.add_argument('--model_name', type=str, default='resnet', 
                        choices=['resnet', 'alexnet', 'googlenet', 'all'], help='Model classes to train and validation, "all" means all possible models')
    parser.add_argument('--scratch', action='store_true', help='Not to use pretrained weights and to train models from scratches')
    parser.add_argument('--epochs', type=int, default=20, help='Maximum epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of samples in a batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

def set_seed(seed):
    """Random seed setting for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    
    # 1. Base resize (applied to all conditions)
    transform_list = [transforms.Resize((224, 224))]

    # 2. Conditional augmentations
    if args.augment == "flip":
        transform_list.append(transforms.RandomHorizontalFlip())
    elif args.augment == "rotation":
        transform_list.append(transforms.RandomRotation(15))
    elif args.augment == "crop":
        transform_list.append(transforms.RandomResizedCrop(224))
    elif args.augment == "all":
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224)
        ])
    elif args.augment == "none":
        pass  # Do nothing
    else:
        raise ValueError(f"Unsupported augmentation option: {args.augment}")

    # 3. Tensor conversion and Normalization
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Pass the list to Compose
    transform = transforms.Compose(transform_list)
    print(f"{args.augment} is applied for data augmentation.")

    dataset = load_data(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir + "_" + args.augment,
        transform=transform,
        use_saved=args.use_saved,
        save_processed=args.save_processed
    )    
    split_ratio = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    loaders = get_loaders(
        dataset, 
        split_ratio=split_ratio, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    class_names = ['Normal', 'Viral Pneumonia', 'COVID-19']
    num_classes = len(class_names)
    
    if args.model_name == 'all':
        models_to_run = ['resnet', 'alexnet', 'googlenet']
        modes = [False, True]  # False = Pretrained, True = Scratch
        runs = [(m, s) for m in models_to_run for s in modes]
    else:
        runs = [(args.model_name, args.scratch)]

    print(f"[*] Total {len(runs)} number of training sessions will run sequentially.\n" + "="*50)

    for run_idx, (model_name, is_scratch) in enumerate(runs, 1):
        pretrained = not is_scratch
        mode_str = "scratch" if is_scratch else "pretrained"
        mode_str = mode_str + f"_{args.augment}"
        run_name = f"{model_name}_{mode_str}"
        
        # Setting seperated directories for each models
        current_save_dir = os.path.join(args.results_dir, run_name)
        os.makedirs(current_save_dir, exist_ok=True)
        
        print(f"\n[{run_idx}/{len(runs)}] 🚀 Session Start: {run_name.upper()}")
        print(f" - model_name: {model_name}")
        print(f" - Pretrained: {pretrained}")
        print(f" - Results Path: {current_save_dir}")
        
        model = MedicalImageClassifier(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
        
        # 1st convolutional layer selection logic to extract feature maps
        feature_layer = None
        if model_name == 'resnet':
            feature_layer = model.model.conv1
        elif model_name == 'alexnet':
            feature_layer = model.model.features[0]
        elif model_name == 'googlenet':
            feature_layer = model.model.conv1.conv
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        trainer = Trainer(
            model=model,
            train_loader=loaders['train_loader'],
            val_loader=loaders['val_loader'],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            class_names=class_names,
            num_epochs=args.epochs,
            save_dir=current_save_dir,
            feature_layer=feature_layer
        )
        
        trainer.fit()
        print(f"[{run_idx}/{len(runs)}] ✅ Session completed: {run_name.upper()}")
        print("="*50)
        
    print("[*] Every sessions is now finished.")

if __name__ == '__main__':
    main()
