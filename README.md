# CellSighter

CellSighter is a deep learning project for automated tumor cell classification from H&E stained histopathology images using multiplex immunofluorescence (mIF) microscopy annotations as ground truth.

## Project Overview

This project implements an advanced pipeline for tumor cell classification that combines:

1. **Feature Extraction**: Utilizes a pretrained vision foundation model to extract patch-wise features from H&E stained histopathology images.

2. **Spatial Context Modeling**: Employs Graph Neural Networks (GNN) to:
   - Structure and aggregate patch embeddings
   - Model cell-cell interactions
   - Capture region-level relationships across tissue samples

3. **Representation Learning**: Implements contrastive learning techniques to:
   - Refine latent space representations
   - Enhance separation between different cell types
   - Improve discriminative power across tumor phenotypes

## Project Structure

```
CellSighter/
├── datasets/
│   └── CellTypes/
│       ├── data/
│       │   └── images/     # Raw images (HxWxC, npz/tiff format)
│       ├── cells/         # Cell segmentation masks (HxW, npz/tiff format)
│       └── cells2labels/  # Cell labels (npz/txt format)
├── src/            # Source code
├── notebooks/      # Jupyter notebooks for experiments
├── configs/        # Configuration files
└── results/        # Model outputs and evaluations
```

### Data Structure Details

The `datasets/CellTypes/` directory follows a specific structure for organizing image data, segmentation masks, and labels:

1. **Raw Images** (`data/images/`):
   - Format: `.npz` or `.tiff`
   - Shape: Height × Width × Channels (HxWxC)
   - C represents the number of protein channels

2. **Cell Segmentation** (`cells/`):
   - Format: `.npz` or `.tiff`
   - Shape: Height × Width (HxW)
   - Contains labeled object matrix where each cell has a unique ID (1 to N)

3. **Cell Labels** (`cells2labels/`):
   - Format: `.npz` or `.txt`
   - Each row corresponds to a cell ID
   - Unlabeled cells (e.g., test data) are marked as -1

  ```

### Training the Binary Cell Classification Model

CellSighter provides a complete pipeline for binary tumor vs non-tumor cell classification. Use the `train_classifier.py` script for easy training:

#### Basic Usage

```bash
# Train with default settings (CNN model, 40 epochs)
python train_classifier.py

# Train with ResNet architecture
python train_classifier.py --model resnet

# Enable data augmentation for better performance
python train_classifier.py --augment
```

#### Advanced Usage

```bash
# Custom training parameters
python train_classifier.py \
    --model resnet \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.0005 \
    --augment

# Use custom configuration file
python train_classifier.py --config path/to/your/config.json

# Resume training from checkpoint
python train_classifier.py --resume checkpoints/best_model.pth
```

#### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `cnn` | Model architecture (`cnn` or `resnet`) |
| `--config` | `src/config.json` | Path to configuration file |
| `--epochs` | - | Number of training epochs (overrides config) |
| `--batch-size` | - | Batch size for training (overrides config) |
| `--lr` | - | Learning rate (overrides config) |
| `--save-dir` | - | Directory to save checkpoints and results |
| `--augment` | - | Enable data augmentation |
| `--no-augment` | - | Disable data augmentation |
| `--resume` | - | Path to checkpoint to resume from |
| `--device` | `auto` | Device to use (`auto`, `cuda`, or `cpu`) |
| `--verbose` | - | Enable verbose output |

#### Model Architectures

**CNN Model (Default)**:
- 4 convolutional blocks with batch normalization
- Global average pooling + fully connected classifier
- ~2M parameters, faster training
- Good for simpler datasets

**ResNet Model**:
- ResNet-inspired with residual connections
- Better gradient flow for deeper networks
- ~11M parameters, more robust
- Better for complex datasets

#### Training Output

Training produces the following files in the checkpoint directory:
- `best_model.pth` - Best performing model
- `training_history.json` - Training metrics over time
- `training_curves.png` - Loss and accuracy plots
- `evaluation_results.json` - Detailed evaluation metrics
- `config.json` - Copy of training configuration

#### Configuration

Training parameters are configured in `src/config.json`:

```json
{
    "crop_size": 128,           
    "num_classes": 2,           
    "epoch_max": 40,            
    "lr": 0.001,                
    "batch_size": 64,           
    "num_workers": 6,           
    "aug": false,               
    "train_set": ["reg001_A", "reg002_A", ...],
    "val_set": ["reg065_A", "reg066_A", ...]
}
```

## License

[Coming soon]