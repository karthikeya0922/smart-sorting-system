"""
Configuration file for model training.
Contains all hyperparameters and settings.
"""

# Model Configuration
MODEL_CONFIG = {
    'architecture': 'mobilenet_v2',  # Options: mobilenet_v2, resnet18, efficientnet_b0
    'num_classes': 6,
    'pretrained': True,  # Use ImageNet pretrained weights
    'dropout': 0.2,
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'optimizer': 'adam',  # Options: adam, sgd, adamw
    'momentum': 0.9,  # For SGD only
}

# Learning Rate Scheduler
SCHEDULER_CONFIG = {
    'type': 'step',  # Options: step, cosine, reduce_on_plateau
    'step_size': 7,  # For StepLR
    'gamma': 0.1,  # For StepLR
    'patience': 3,  # For ReduceLROnPlateau
    'min_lr': 1e-6,  # Minimum learning rate
}

# Data Configuration
DATA_CONFIG = {
    'image_size': 224,
    'num_workers': 2,  # Number of data loading workers (reduce if causing issues)
    'pin_memory': False,  # Set to True if using GPU
    'train_dir': 'dataset/processed/train',
    'val_dir': 'dataset/processed/val',
    'test_dir': 'dataset/processed/test',
}

# Checkpoint Configuration
CHECKPOINT_CONFIG = {
    'save_dir': 'model/checkpoints',
    'save_every': 5,  # Save checkpoint every N epochs
    'save_best_only': True,  # Only save the best model
    'metric': 'val_acc',  # Metric to determine best model: val_acc or val_loss
}

# Early Stopping
EARLY_STOPPING_CONFIG = {
    'enabled': True,
    'patience': 7,  # Stop if no improvement for N epochs
    'min_delta': 0.001,  # Minimum change to qualify as improvement
}

# Class Names (in order)
CLASS_NAMES = [
    'glass',
    'metal',
    'non-recyclable',
    'organic',
    'paper',
    'plastic'
]

# Class weights (for imbalanced datasets, set to None for equal weights)
CLASS_WEIGHTS = None  # Will be calculated automatically if None

# Device Configuration
DEVICE_CONFIG = {
    'use_gpu': True,  # Automatically fall back to CPU if GPU not available
    'gpu_id': 0,  # GPU device ID
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_file': 'model/training.log',
    'print_frequency': 10,  # Print every N batches
    'save_plots': True,
}

# Paths
PATHS = {
    'train_dir': DATA_CONFIG['train_dir'],
    'val_dir': DATA_CONFIG['val_dir'],
    'test_dir': DATA_CONFIG['test_dir'],
    'checkpoint_dir': CHECKPOINT_CONFIG['save_dir'],
    'results_dir': 'model/results',
}
