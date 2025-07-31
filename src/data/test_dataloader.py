import unittest
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from .utils import load_samples

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Load configuration from config.json
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Use only first image for testing
        self.config['train_set'] = [self.config['train_set'][0]]
        
        # Load samples using the load_samples function
        self.crops = load_samples(self.config, self.config['train_set'])

    def test_data_loading_and_visualization(self):
        """Test loading real data and visualizing samples"""
        # Create output directory in data folder
        test_output_dir = os.path.join(os.path.dirname(__file__), 'test_outputs')
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Separate crops by label
        label_0_crops = [crop for crop in self.crops if crop._label == 0]
        label_1_crops = [crop for crop in self.crops if crop._label != 0]
        
        print(f"\nFound {len(label_0_crops)} samples with label 0 ({self.config['hierarchy_match']['0']})")
        print(f"Found {len(label_1_crops)} samples with label 1 ({self.config['hierarchy_match']['1']})")
        
        # Take up to 3 samples from each class
        num_per_class = min(3, min(len(label_0_crops), len(label_1_crops)))
        selected_crops = (label_0_crops[:num_per_class] + 
                        label_1_crops[:num_per_class])
        
        # Create a figure with subplots
        num_samples = len(selected_crops)
        fig, axes = plt.subplots(num_samples, 1, figsize=(8, 4*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for idx, crop in enumerate(selected_crops):
            # Get sample from crop
            sample = crop.sample(mask=False)  # We only need image and label
            
            # Plot original image (take mean across channels if more than 3 channels)
            image = sample['image']
            if image.shape[2] > 3:
                display_image = np.mean(image, axis=2)
            else:
                display_image = image
                
            # Normalize image for better visualization
            display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min())
            
            # Plot image
            axes[idx].imshow(display_image, cmap='gray')
            label_name = self.config['hierarchy_match'][str(sample['label'])]
            axes[idx].set_title(f"Cell {sample['cell_id']} from {sample['image_id']}\n"
                              f"Label: {sample['label']} ({label_name})")
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_file = os.path.join(test_output_dir, 'balanced_cell_samples.png')
        plt.savefig(output_file)
        plt.close()
        
        # Print information about the samples
        print(f"\nVisualized {num_samples} samples ({num_per_class} from each class):")
        for idx, crop in enumerate(selected_crops):
            sample = crop.sample(mask=False)
            print(f"Sample {idx+1}:")
            print(f"  Image ID: {sample['image_id']}")
            print(f"  Cell ID: {sample['cell_id']}")
            print(f"  Label: {sample['label']} ({self.config['hierarchy_match'][str(sample['label'])]})")
            print(f"  Image shape: {sample['image'].shape}")
            print()

        # Basic assertions
        self.assertGreater(len(self.crops), 0, "No crops were loaded")
        self.assertIsNotNone(self.crops[0], "First crop is None")
        self.assertTrue(any(crop._label == 0 for crop in self.crops), "No samples with label 0")
        self.assertTrue(any(crop._label != 0 for crop in self.crops), "No samples with label 1")
        self.assertTrue(os.path.exists(output_file), "Visualization file was not created")

if __name__ == '__main__':
    unittest.main()

# run this file with: python -m unittest data/test_dataloader.py -v