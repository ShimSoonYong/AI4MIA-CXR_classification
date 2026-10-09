import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split

# configuration variables
DATA_ROOT = "dataset"
RAW_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")

class ProcessedDataset(Dataset):
    """Wrapper class to read saved .pt tensor dataset"""
    def __init__(self, data_dict):
        self.images = data_dict['images']
        self.targets = data_dict['labels']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def load_data(raw_dir: str, processed_dir: str, transform=None, use_saved: bool = True, save_processed: bool = False):
    """
    main function to load dataset
    - use_saved=True: If saved .pt file exists, than load that.
    - save_processed=True: When load raw data, transform and save them to the .pt tensors to load faster later.
    """
    processed_path = os.path.join(processed_dir, "processed_dataset.pt")

    if use_saved and os.path.exists(processed_path):
        print(f"---Loading saved .pt tensor dataset: {processed_path} ---")
        data = torch.load(processed_path)
        return ProcessedDataset(data)

    print(f"--- Loading raw dataset: {raw_dir} ---")
    dataset = datasets.ImageFolder(root=raw_dir, transform=transform)
    
    print("Remapping class labels")
    orig_idx_covid = dataset.class_to_idx['COVID']
    orig_idx_normal = dataset.class_to_idx['Normal']
    orig_idx_viral = dataset.class_to_idx['Viral Pneumonia']
    
    mapping = {
        orig_idx_normal: 0,
        orig_idx_viral: 1,
        orig_idx_covid: 2
    }
    
    dataset.targets = [mapping[t] for t in dataset.targets]
    dataset.samples = [(path, mapping[t]) for path, t in dataset.samples]
    dataset.classes = ['Normal', 'Viral pneumonia', 'COVID-19']
    dataset.class_to_idx = {'Normal': 0, 'Viral pneumonia': 1, 'COVID-19': 2}

    if save_processed:
        print(f"--- Saving processed .pt data at {processed_dir} folder... ---")
        os.makedirs(processed_dir, exist_ok=True)
        
        images, labels = [], []
        for img, label in dataset:
            images.append(img)
            labels.append(label)
            
        torch.save({
            'images': torch.stack(images),
            'labels': torch.tensor(labels)
        }, processed_path) 
        print(f".pt tensor dataset saved at: {processed_path}")
    else:
        print("Dataset will be processed by DataLoader")
        
    return dataset


def get_loaders(dataset, split_ratio: dict = None, batch_size: int = 32, num_workers: int = 4):
    """
    Stratified Split for class imbalance and instantiate DataLoader
    """
    if split_ratio is None:
        split_ratio = {'train': 0.8, 'val': 0.2, 'test': 0.0}

    print(f"Dataset split ratio: {split_ratio}")
    print(f"Batch Size: {batch_size}, Num Workers: {num_workers}")
    
    targets = np.array(dataset.targets)
    indices = np.arange(len(dataset))
    
    train_ratio = split_ratio.get('train', 0.8)
    val_ratio = split_ratio.get('val', 0.2)
    test_ratio = split_ratio.get('test', 0.0)

    try:
        if test_ratio > 0.0:
            train_idx, temp_idx, _, temp_targets = train_test_split(
                indices, targets,
                stratify=targets,
                test_size=(val_ratio + test_ratio),
                random_state=42
            )
            val_idx, test_idx = train_test_split(
                temp_idx,
                stratify=temp_targets,
                test_size=(test_ratio / (val_ratio + test_ratio)),
                random_state=42
            )
        else:
            train_idx, val_idx = train_test_split(
                indices,
                stratify=targets,
                test_size=val_ratio,
                random_state=42
            )
            test_idx = []

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx) if len(test_idx) > 0 else None

        print(f"Splitted as: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset) if test_dataset else 0}")

        loaders = {
            "train_loader": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            "val_loader": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        }
        
        if test_dataset:
            loaders["test_loader"] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        return loaders

    except Exception as e:
        print(f"Error during dataset split: {e}")
        raise RuntimeError("Failed to instantiate Stratified DataLoader")


if __name__ == '__main__':
    dataset = load_data(
        raw_dir=RAW_DIR, 
        processed_dir=PROCESSED_DIR, 
        transform=transforms.ToTensor(), 
        use_saved=True,
        save_processed=True
    )
    
    loaders = get_loaders(dataset, num_workers=5)
    
    train_batch = next(iter(loaders["train_loader"]))
    print(f"\n[Success] Train Batch Image Shape: {train_batch[0].shape}")
    print(f"[Success] Train Batch Label Shape: {train_batch[1].shape}")
