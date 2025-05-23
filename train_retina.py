import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from retina_data_module import RetinaDataModule
from retina_model import LitRetinaNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data',
                      help='Directory containing the retina dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--epochs', type=int, default=25,
                      help='Number of epochs to train')
    parser.add_argument('--debug', action='store_true',
                      help='Run in debug mode with limited batches')
    parser.add_argument('--evaluate', action='store_true',
                      help='Run evaluation after training')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/checkpoints', exist_ok=True)
    
    # Initialize data module
    dm = RetinaDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Initialize model
    model = LitRetinaNet(lr=args.lr)
    
    # Attach the data module to the model
    model.datamodule = dm
    
    # Define callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath='models/checkpoints',
        filename='best_model',
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )
    
    early_stopping_cb = EarlyStopping(
        monitor='val_acc',
        patience=8,
        mode='max',
        verbose=True
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stopping_cb],
        limit_train_batches=30 if args.debug else 1.0,
        limit_val_batches=30 if args.debug else 1.0,
        deterministic=True,
        enable_progress_bar=True,
    )
    
    # Train the model
    print(trainer.accelerator)
    trainer.fit(model, dm)
    
    # Evaluate if requested
    if args.evaluate:
        print("\nEvaluation started...")
        test_results = trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path='best')
        print("\nTest Results:", test_results) 