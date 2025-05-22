import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lit_model import LitResNet
from data_module import CIFAR10DataModule
from plotting import plot_training_curves
from settings import DATA_DIR, BATCH_SIZE, LR, MAX_EPOCHS, MODEL_DIR, CHECKPOINT_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet model on CIFAR-10")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode (fewer batches)")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=LR, help="learning rate")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Initialize data module
    dm = CIFAR10DataModule(
        data_path=DATA_DIR,
        batch_size=BATCH_SIZE
    )

    dm.setup("fit")
    
    logger = pl.loggers.CSVLogger(save_dir=MODEL_DIR, name='logs')
    
    # Initialize model with 10 classes for CIFAR-10
    model = LitResNet(num_classes=10, lr=args.lr)
    
    # Attach the data module to the model so that GradCAM can access normalization stats
    model.datamodule = dm

    # Define callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
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
    
    # Trainer with debug mode
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stopping_cb],
        limit_train_batches=30 if args.debug else 1.0,
        limit_val_batches=30 if args.debug else 1.0,
        deterministic=True, 
        enable_progress_bar=False, 
    )

    # Training
    trainer.fit(model, dm)

    # Evaluation (if requested)
    if args.evaluate:
        print("\nEvaluation started...")
        test_results = trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path='best')
        print("\nTest Results:", test_results)



