import argparse
import os
import yaml
import importlib
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import models, transforms
import torch.nn as nn
from data_module import DataModule
from lit_model import LitModel
import datasets

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_transforms(transform_config):
    """Build transforms from config."""
    transform_list = []
    for t in transform_config:
        name = t['name']
        params = t.get('params', {})
        transform_class = getattr(transforms, name)
        transform_list.append(transform_class(**params))
    return transforms.Compose(transform_list)

def get_dataset_class(dataset_name):
    """Get dataset class from datasets.py."""
    return getattr(datasets, dataset_name)

def get_model(model_config):
    """Get model based on configuration."""
    name = model_config['name'].lower()
    source = model_config['source'].lower()
    num_classes = model_config['num_classes']
    
    if source == 'torchvision':
        # Get the model from torchvision
        model_fn = getattr(models, name)
        model = model_fn(pretrained=model_config.get('pretrained', True))
        
        # Modify the last layer for our number of classes
        if name.startswith('resnet'):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif name == 'alexnet':
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        elif name.startswith('vgg'):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif source == 'custom':
        # Import custom model from a separate file
        try:
            custom_models = importlib.import_module('custom_models')
            model_class = getattr(custom_models, name)
            model = model_class(num_classes=num_classes)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Custom model {name} not found: {e}")
    else:
        raise ValueError(f"Unknown model source: {source}")
    
    return model

def get_datamodule(data_config):
    """Get datamodule based on configuration."""
    # Get dataset classes
    train_val_dataset_class = get_dataset_class(data_config['dataset']['train_val_dataset'])
    test_dataset_class = get_dataset_class(data_config['dataset']['test_dataset'])
    
    # Build transforms
    train_transforms = None
    val_test_transforms = None
    if 'transforms' in  data_config['dataset']:
        if 'train' in data_config['dataset']['transforms']:
            train_transforms = build_transforms(data_config['dataset']['transforms']['train'])
            
        if 'val_test' in data_config['dataset']['transforms']:
            val_test_transforms = build_transforms(data_config['dataset']['transforms']['val_test'])
    
    # Create datasets
    dataset_params = data_config['dataset']['params']
    train_val_dataset = train_val_dataset_class(
        train=True,
        **dataset_params
    )
    test_dataset = test_dataset_class(
        train=False,
        **dataset_params
    )
    
    # Create datamodule
    return DataModule(
        train_val_dataset=train_val_dataset,
        test_dataset=test_dataset,
        train_transforms=train_transforms,
        val_test_transforms=val_test_transforms,
        batch_size=data_config['datamodule']['batch_size'],
        num_workers=data_config['datamodule']['num_workers'],
        split_ratio=data_config['datamodule']['split_ratio']
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with YAML config")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Create directories if they don't exist
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Initialize data module
    dm = get_datamodule(config['data'])
    dm.setup("fit")
    
    # Initialize logger
    logger = pl.loggers.CSVLogger(
        save_dir=config['paths']['log_dir'],
        name='logs'
    )
    
    # Initialize model
    model = get_model(config['model'])
    if config['model']['load_checkpoint']:
        lit_model = LitModel.load_from_checkpoint(
            config['paths']['checkpoint_dir'] + "/best_model.ckpt",
            model=model,
            num_classes=config["model"]["num_classes"],
            **config["lit_model"]
        )
    else:
        lit_model = LitModel(
            model=model,
            num_classes=config["model"]["num_classes"],
            **config["lit_model"]
        )
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config['paths']['checkpoint_dir'],
            filename='best_model',
            monitor='val_acc',
            mode='max',
            save_top_k=1
        ),
        EarlyStopping(
            monitor='val_acc',
            patience=config['training']['early_stopping_patience'],
            mode='max',
            verbose=True
        )
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        logger=logger,
        max_epochs=config['training']['max_epochs'],
        callbacks=callbacks,
        limit_train_batches=30 if config['training']['debug'] else 1.0,
        limit_val_batches=30 if config['training']['debug'] else 1.0,
        deterministic=config['trainer']['deterministic'],
        enable_progress_bar=config['trainer']['enable_progress_bar']
    )
    
    # Training
    trainer.fit(lit_model, dm)
    
    # Evaluation (if requested)
    if config['training']['evaluate']:
        print("\nEvaluation started...")
        test_results = trainer.test(
            lit_model,
            dataloaders=dm.test_dataloader(),
            ckpt_path='best'
        )
        print("\nTest Results:", test_results)

if __name__ == '__main__':
    main()
