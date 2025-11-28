import cv2
from deepface import DeepFace
import sys
import os
import numpy as np
from tqdm import tqdm

def detect_expressions_in_video(video_path, output_path=None, display=False):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pbar = tqdm(total=frame_count, desc="Processing video frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Analisa o frame detectando múltiplas faces
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')

            # Garante que results seja sempre uma lista
            if not isinstance(results, list):
                results = [results]

            # Processa cada face detectada
            for face_data in results:
                # Obtém a região da face
                face_region = face_data.get('region', {})
                x = face_region.get('x', 0)
                y = face_region.get('y', 0)
                w = face_region.get('w', 0)
                h = face_region.get('h', 0)

                # Obtém a emoção dominante
                dominant_emotion = face_data['dominant_emotion']
                emotion_confidence = face_data['emotion'][dominant_emotion]

                # Desenha retângulo ao redor da face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Adiciona o texto da emoção acima da face
                label = f"{dominant_emotion} ({emotion_confidence:.1f}%)"
                label_y = max(y - 10, 20)
                cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        except Exception as e:
            print(f"Warning: {e}")

        if display:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if output_path:
            out.write(frame)

        pbar.update(1)

    pbar.close()
    cap.release()
    if output_path:
        out.release()
    if display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_file = os.path.join(script_dir, "videos/input_video.mp4")
    output_file = os.path.join(script_dir, "videos/output_video.mp4")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    detect_expressions_in_video(video_file, output_file, display=False)