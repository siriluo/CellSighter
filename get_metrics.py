#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path for imports
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

from evaluate import main as eval_main

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train binary cell classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_metrics.py
  python get_metrics.py --model_path src/model.pth
        """
    )
    
    # Model and training arguments
    parser.add_argument('--config', type=str, default='src/config.json',
                    help='Path to configuration file (default: src/config.json)')

    parser.add_argument('--model_path', type=str, default='src/model.pth',
                    help='Path to saved model checkpoint file (default: src/model.pth)')
    
    parser.add_argument('--model', type=str, default='cnn', 
                    choices=['cnn', 'resnet'],
                    help='Model architecture to use (default: cnn)')
    
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
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Please make sure the config file exists or specify a valid path.")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"Error: Checkpoint file not found: {args.model_path}")
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
        config['save_dir'] = './eval_metrics'
    
    # Save updated config to temporary file
    temp_config_path = 'temp_config.json'
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return temp_config_path


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    validate_arguments(args)
    
    # Update config with command line overrides
    temp_config_path = update_config_from_args(args.config, args)
    
    try:
        
        # Set device environment variable if specified
        if args.device != 'auto':
            if args.device == 'cpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            elif args.device == 'cuda':
                # Let PyTorch auto-detect CUDA devices
                pass
        
        # Run training
        metrics_evaluator, history = eval_main(
            config_path=temp_config_path,
            checkpoint_path=args.model_path,
            model_type=args.model,
        )


        
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



