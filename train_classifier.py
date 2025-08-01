#!/usr/bin/env python3
"""
train_classifier.py - Main Entry Point for Cell Classification Training

This script provides a simple interface to train the binary cell classification model.
It handles command-line arguments and calls the main training pipeline.

Usage:
    python train_classifier.py                    # Use default settings
    python train_classifier.py --model resnet     # Use ResNet architecture
    python train_classifier.py --config custom_config.json  # Use custom config
    python train_classifier.py --resume checkpoints/best_model.pth  # Resume training
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path for imports
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

# Import the main training function
from train import main as train_main


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train binary cell classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_classifier.py
  python train_classifier.py --model resnet --config src/config.json
  python train_classifier.py --resume checkpoints/best_model.pth
  python train_classifier.py --model cnn --epochs 50 --lr 0.0005
        """
    )
    
    # Model and training arguments
    parser.add_argument('--config', type=str, default='src/config.json',
                       help='Path to configuration file (default: src/config.json)')
    
    parser.add_argument('--model', type=str, default='cnn', 
                       choices=['cnn', 'resnet'],
                       help='Model architecture to use (default: cnn)')
    
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    # Override configuration parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training (overrides config)')
    
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save checkpoints and results')
    
    # Data augmentation
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation')
    
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation')
    
    # Device selection
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training (default: auto)')
    
    # Verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Please make sure the config file exists or specify a valid path.")
        sys.exit(1)
    
    # Check resume checkpoint
    if args.resume and not os.path.exists(args.resume):
        print(f"Error: Checkpoint file not found: {args.resume}")
        sys.exit(1)
    
    # Check conflicting augmentation arguments
    if args.augment and args.no_augment:
        print("Error: Cannot specify both --augment and --no-augment")
        sys.exit(1)
    
    # Validate save directory
    if args.save_dir:
        try:
            os.makedirs(args.save_dir, exist_ok=True)
        except Exception as e:
            print(f"Error: Cannot create save directory {args.save_dir}: {e}")
            sys.exit(1)


def update_config_from_args(config_path, args):
    """Update configuration with command line arguments."""
    import json
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override with command line arguments
    if args.epochs is not None:
        config['epoch_max'] = args.epochs
        if args.verbose:
            print(f"Override: epochs = {args.epochs}")
    
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
        if args.verbose:
            print(f"Override: batch_size = {args.batch_size}")
    
    if args.lr is not None:
        config['lr'] = args.lr
        if args.verbose:
            print(f"Override: learning_rate = {args.lr}")
    
    if args.save_dir is not None:
        config['save_dir'] = args.save_dir
        if args.verbose:
            print(f"Override: save_dir = {args.save_dir}")
    
    # Handle augmentation
    if args.augment:
        config['aug'] = True
        if args.verbose:
            print("Override: data augmentation enabled")
    elif args.no_augment:
        config['aug'] = False
        if args.verbose:
            print("Override: data augmentation disabled")
    
    # Set default save directory if not specified
    if 'save_dir' not in config:
        config['save_dir'] = './checkpoints'
    
    # Save updated config to temporary file
    temp_config_path = 'temp_config.json'
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return temp_config_path


def print_training_info(args, config_path):
    """Print training information."""
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("=" * 60)
    print("CELL CLASSIFICATION TRAINING")
    print("=" * 60)
    print(f"Model architecture: {args.model.upper()}")
    print(f"Configuration file: {args.config}")
    print(f"Training epochs: {config['epoch_max']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Data augmentation: {'Enabled' if config.get('aug', False) else 'Disabled'}")
    print(f"Save directory: {config['save_dir']}")
    
    if args.resume:
        print(f"Resuming from: {args.resume}")
    
    print(f"Training images: {len(config['train_set'])}")
    print(f"Validation images: {len(config['val_set'])}")
    print("=" * 60)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    validate_arguments(args)
    
    # Update config with command line overrides
    temp_config_path = update_config_from_args(args.config, args)
    
    try:
        # Print training information
        if args.verbose:
            print_training_info(args, temp_config_path)
        
        # Set device environment variable if specified
        if args.device != 'auto':
            if args.device == 'cpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            elif args.device == 'cuda':
                # Let PyTorch auto-detect CUDA devices
                pass
        
        # Run training
        print("Starting training...")
        trainer, history, eval_results = train_main(
            config_path=temp_config_path,
            model_type=args.model,
            resume_checkpoint=args.resume
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Print summary
        print(f"Best F1 Score: {trainer.best_val_f1:.4f}")
        print(f"Best Accuracy: {trainer.best_val_acc:.4f}")
        print(f"Final AUC: {eval_results['auc']:.4f}")
        
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return 1
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
        
    finally:
        # Clean up temporary config
        if os.path.exists('temp_config.json'):
            os.remove('temp_config.json')


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)