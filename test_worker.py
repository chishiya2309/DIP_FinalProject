import argparse
import time
import cv2

from core.inference_worker import FallDetectorWorker

def main():
    parser = argparse.ArgumentParser(description="Test FallDetectorWorker standalone")
    parser.add_argument("--video", type=str, default="data/sample_fall.mp4", help="Path to input video")
    parser.add_argument("--model-path", type=str, default="train_hung/runs/fall_detection/best.pt", help="Path to trained PoseBiGRU model")
    parser.add_argument("--yolo-model", type=str, default="yolov8n-pose.pt", help="Path to YOLOv8-pose model")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Fall confidence threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device to run inference on (auto, cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    print(f"Initializing Worker...")
    print(f"Video source: {args.video}")
    
    worker = FallDetectorWorker(
        model_path=args.model_path,
        yolo_model=args.yolo_model,
        device=args.device,
        conf_threshold=args.conf_threshold,
    )
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.video}")
        return
        
    frame_idx = 0
    start_time = time.time()
    
    print("Starting inference. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
            
        current_time = time.time() - start_time
        frame_idx += 1
        
        # We simulate the exact logic that CameraManager would do
        annotated_frame, fall_probs = worker.process_frame(frame, current_time)
        
        if fall_probs:
            probs_str = ", ".join([f"ID {tid}: {p:.2f}" for tid, p in fall_probs.items()])
            print(f"Frame {frame_idx} | {probs_str}")
        
        # Display the output
        cv2.imshow("Test FallDetectorWorker", annotated_frame)
        
        # Wait 1 ms, quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
