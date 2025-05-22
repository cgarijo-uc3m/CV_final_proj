import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(metrics_path="training_metrics.npz", output_path="training_curves.png"):
    """Plots training and validation accuracy curves"""
    # Load saved metrics
    metrics = np.load(metrics_path)
    train_acc = metrics['train_acc']
    val_acc = metrics['val_acc']
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    print(f"Saved training curves to {output_path}")
