import json
import matplotlib.pyplot as plt
import os

def plot_loss(json_path, output_path):
    if not os.path.exists(json_path):
        print(f"File không tồn tại: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    epochs = [item['epoch'] for item in data]
    train_loss = [item['train_loss'] for item in data]
    val_loss = [item['val_loss'] for item in data]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o', linewidth=2, color='#1f77b4')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='s', linewidth=2, color='#ff7f0e')

    plt.title('Training and Validation Loss over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(epochs)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Đã lưu đồ thị thành công tại: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    json_path = os.path.join(base_dir, 'outputs', 'result_train_yolo11m-pose', 'history.json')
    output_path = os.path.join(base_dir, 'docs', 'Loss_Curve.png')
    
    plot_loss(json_path, output_path)
