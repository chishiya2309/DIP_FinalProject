import numpy as np
import matplotlib.pyplot as plt
import os

def plot_cm(tp, fp, fn, tn, output_path):
    cm = np.array([[tn, fp], [fn, tp]])
    labels = ['Non-Fall', 'Fall']
    
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    # Đặt nhãn cho trục x và y
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.xaxis.set_ticks_position('bottom')
    
    # Hiển thị số liệu lên ma trận
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > (cm.max() / 2) else "black"
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=16, fontweight='bold', color=color)

    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)
    plt.title('Confusion Matrix (Threshold = 0.73)', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Đã lưu Confusion Matrix tại: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_path = os.path.join(base_dir, 'docs', 'Confusion_Matrix.png')
    
    # Values extracted from threshold_report.csv at optimal threshold 0.73
    tp = 309
    fp = 37
    fn = 43
    tn = 16277
    
    plot_cm(tp, fp, fn, tn, output_path)
