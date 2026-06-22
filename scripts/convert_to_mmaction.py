"""
Script convert dữ liệu keypoints JSON sang định dạng .pkl cho MMAction2 / PoseC3D.
Áp dụng Sliding Window và mapping nhãn (frame-level) chuẩn xác để tránh nhiễu data.

Định dạng output PoseC3D (mỗi sample là 1 dictionary):
{
    'frame_dir': str (VD: 'fall-01-cam0_win001'),
    'label': int (0: ADL, 1: FALL),
    'img_shape': tuple (height, width),
    'original_shape': tuple (height, width),
    'total_frames': int (VD: 60),
    'keypoint': np.ndarray, shape [M, T, V, 2]
    'keypoint_score': np.ndarray, shape [M, T, V]
}
"""

import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm

WINDOW_SIZE = 30
STRIDE = 15
MAX_PERSONS = 1
NUM_KEYPOINTS = 17

def load_ur_fall_frame_labels(data_raw_dir):
    """Doc file CSV cua UR Fall de map tung frame voi state (-1, 0, 1)"""
    labels = {}
    ur_dir = os.path.join(data_raw_dir, "ur_fall_detection")
    csv_falls = os.path.join(ur_dir, "urfall-cam0-falls.csv")
    csv_adls = os.path.join(ur_dir, "urfall-cam0-adls.csv")

    for csv_file in [csv_falls, csv_adls]:
        if not os.path.exists(csv_file):
            print(f"[Canh bao] Khong tim thay CSV UR Fall: {csv_file}")
            continue
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    vid_name = parts[0]
                    # UR Fall JSON dang luu voi duoi -cam0
                    vid_full = f"{vid_name}-cam0"
                    frame_idx = int(parts[1]) - 1 # Chuyen 1-based ve 0-based index
                    state = int(parts[2])
                    
                    if vid_full not in labels:
                        labels[vid_full] = {}
                    labels[vid_full][frame_idx] = state
    return labels

def process_video(json_path, ur_labels_dict):
    """Xu ly 1 file JSON va cat thanh nhieu sample (windows)"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    video_name = data.get("video_name", os.path.basename(json_path).replace('.json', ''))
    source = data.get("source", "unknown")
    img_shape = (data.get("resolution", [480, 640])[1], data.get("resolution", [480, 640])[0]) # H, W
    frames_data = data.get("frames", [])
    
    if len(frames_data) == 0:
        return []

    # Map data frame vao dictionary de lookup nhanh O(1)
    frames_dict = {fd["frame_idx"]: fd["persons"] for fd in frames_data}
    max_frame = max(frames_dict.keys())
    
    samples = []
    
    # Cắt Sliding Window
    for w_start in range(0, max_frame - WINDOW_SIZE + 2, STRIDE):
        w_end = w_start + WINDOW_SIZE - 1
        
        # 1. Xác định Nhãn cho Window
        window_label = 0 # Default: ADL
        
        if source == "ur_fall_detection":
            # Xu ly nhan cho UR Fall frame-level
            if video_name in ur_labels_dict:
                states_in_window = []
                for i in range(w_start, w_end + 1):
                    states_in_window.append(ur_labels_dict[video_name].get(i, -1))
                
                # Nếu window có chứa frame té ngã (0: falling, 1: fallen)
                # Yêu cầu ít nhất 5 frames có trạng thái ngã để xác nhận đây là window FALL
                fall_frames = sum(1 for s in states_in_window if s in [0, 1])
                if fall_frames >= 5:
                    window_label = 1
            else:
                # Fallback neu thieu CSV
                if "fall" in video_name:
                    window_label = 1

        elif source == "multiple_cameras_fall":
            # Xu ly nhan cho MCFD segment-level
            segments = data.get("segments", [])
            for seg in segments:
                if seg["label"] == 1:
                    # Tinh overlap giua window va fall segment
                    overlap_start = max(w_start, seg["start"])
                    overlap_end = min(w_end, seg["end"])
                    overlap_frames = overlap_end - overlap_start + 1
                    
                    if overlap_frames >= 5: # Yêu cầu ít nhất 5 frames nằm trong cú ngã
                        window_label = 1
                        break
        
        # 2. Extract Keypoints
        kp_array = np.zeros((MAX_PERSONS, WINDOW_SIZE, NUM_KEYPOINTS, 2), dtype=np.float32)
        score_array = np.zeros((MAX_PERSONS, WINDOW_SIZE, NUM_KEYPOINTS), dtype=np.float32)
        
        valid_frames = 0
        for i in range(WINDOW_SIZE):
            f_idx = w_start + i
            if f_idx in frames_dict and len(frames_dict[f_idx]) > 0:
                valid_frames += 1
                # Chon nguoi co bounding box confidence cao nhat
                person = sorted(frames_dict[f_idx], key=lambda x: x["bbox_conf"], reverse=True)[0]
                kps = person["keypoints"]
                
                for v in range(NUM_KEYPOINTS):
                    kp_array[0, i, v, 0] = kps[v][0]     # X
                    kp_array[0, i, v, 1] = kps[v][1] # Y
                    score_array[0, i, v] = kps[v][2] # Score
                    
        # Bo qua cac window qua nhieu frame khong co nguoi (duoi 30 frames co nguoi)
        if valid_frames < WINDOW_SIZE // 2:
            continue
            
        sample = {
            'frame_dir': f"{video_name}_win{w_start:04d}",
            'label': window_label,
            'img_shape': img_shape,
            'original_shape': img_shape,
            'total_frames': WINDOW_SIZE,
            'keypoint': kp_array,
            'keypoint_score': score_array
        }
        samples.append(sample)
        
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", default="data/processed/keypoints", help="Thu muc goc chua keypoints JSON")
    parser.add_argument("--raw_dir", default="data/raw", help="Thu muc chua data raw (de doc CSV UR Fall)")
    parser.add_argument("--out_dir", default="data/processed/mmaction", help="Thu muc luu file .pkl")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print("[*] Dang Load CSV labels cua UR Fall...")
    ur_labels_dict = load_ur_fall_frame_labels(args.raw_dir)
    print(f" -> Load duoc labels cua {len(ur_labels_dict)} video UR Fall.")

    all_samples = []

    # Duyet tim tat ca file JSON trong thu muc keypoints (bao gom ca ur_fall va mcfd)
    json_paths = []
    for root, _, files in os.walk(args.json_dir):
        for f in files:
            if f.endswith('.json'):
                json_paths.append(os.path.join(root, f))

    print(f"[*] Bat dau xu ly Sliding Window cho {len(json_paths)} videos...")
    
    for jpath in tqdm(json_paths, desc="Converting"):
        samples = process_video(jpath, ur_labels_dict)
        all_samples.extend(samples)

    print(f"[*] Hoan tat. Tong so samples (clips) duoc tao ra: {len(all_samples)}")
    
    # Chia Train / Val (Tam thoi split random 80/20 de kiem tra)
    # De chuan xac hon ban nen chia theo Cross-Subject hoac theo Chute trong MCFD.
    np.random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    
    train_data = all_samples[:split_idx]
    val_data = all_samples[split_idx:]
    
    train_path = os.path.join(args.out_dir, "train_data.pkl")
    val_path = os.path.join(args.out_dir, "val_data.pkl")
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
        
    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)
        
    print(f"[OK] Da luu thanh cong:")
    print(f"  -> Train: {train_path} ({len(train_data)} samples)")
    print(f"  -> Val  : {val_path} ({len(val_data)} samples)")

if __name__ == "__main__":
    main()
