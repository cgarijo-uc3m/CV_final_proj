import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def generate_classification_report(test_targets, test_preds, output_path="classification_report.txt"):
    """Generates and saves classification report with confusion matrix"""
    # Text report
    report = classification_report(test_targets, test_preds)
    with open(output_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
    
    # Confusion matrix visualization
    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Saved classification report to {output_path}")
    print(f"Saved confusion matrix to confusion_matrix.png")
