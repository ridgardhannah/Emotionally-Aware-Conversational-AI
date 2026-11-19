import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd

#loop through vids. creates csv for each, with muscle and gaze data

#1 setup the AI model
print("Loading PyTorch AU Vision Model (ResNet-18)...")
au_model = models.resnet18(weights=None)
au_model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(au_model.fc.in_features, 12))
#this is the trained model:
au_model.load_state_dict(torch.load("best_vision_model_EXPANDED_FINAL.pth", map_location='cpu', weights_only=True))
au_model.eval()
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

print("Loading MediaPipe Gaze Model...")
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

#folders
RAW_DIR = "raw_videos"
PROC_DIR = "processed_data"
os.makedirs(PROC_DIR, exist_ok=True) #will create the folder if it's missing


#2 process all the videos
au_list = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU14", "AU15", "AU17", "AU23", "AU24"]

for filename in os.listdir(RAW_DIR):
    if filename.endswith(".mp4"):
        video_path = os.path.join(RAW_DIR, filename)
        emotion_name = filename.replace("mini_", "").replace(".mp4", "")
        print(f"\nProcessing: {filename} (Emotion: {emotion_name.upper()})")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps / 5) #5 frames a second
        
        results = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % frame_skip == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                #PyTorch AU extraction
                img_tensor = transform(Image.fromarray(rgb_frame)).unsqueeze(0)
                with torch.no_grad():
                    # We are only getting ONE set of AUs now (not left/right)
                    au_probs = torch.sigmoid(au_model(img_tensor))[0].tolist()
                
                #then MediaPipe gaze extraction
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                mp_results = detector.detect(mp_image)
                
                gaze_v = 0.5
                gaze_h = 0.5
                gaze_dir = "FORWARD"
                
                if mp_results.face_landmarks:
                    landmarks = mp_results.face_landmarks[0]
                    
                    #first calc vertical gaze. y axis
                    left_v = (landmarks[468].y - landmarks[159].y) / (landmarks[145].y - landmarks[159].y + 1e-6)
                    right_v = (landmarks[473].y - landmarks[386].y) / (landmarks[374].y - landmarks[386].y + 1e-6)
                    gaze_v = (left_v + right_v) / 2
                    
                    #then calc horizontal gaze. x axis
                    left_h = (landmarks[468].x - landmarks[33].x) / (landmarks[133].x - landmarks[33].x + 1e-6)
                    right_h = (landmarks[473].x - landmarks[362].x) / (landmarks[263].x - landmarks[362].x + 1e-6)
                    gaze_h = (left_h + right_h) / 2
                    
                    #then use these to categorise mainn gaze direction
                    if gaze_v < 0.40: gaze_dir = "UP"
                    elif gaze_v > 0.60: gaze_dir = "DOWN"
                    elif gaze_h < 0.45: gaze_dir = "RIGHT" #mirrored for the camera
                    elif gaze_h > 0.55: gaze_dir = "LEFT"
                
                #finally save the row
                timestamp = round(frame_count / fps, 2)
                row_data = [timestamp] + au_probs + [gaze_v, gaze_h, gaze_dir]
                results.append(row_data)
                
            frame_count += 1
            
        cap.release()
        
        #save to processed data folder
        columns = ["Time"] + au_list + ["Gaze_V", "Gaze_H", "Gaze_Direction"]        
        df = pd.DataFrame(results, columns=columns)
        
        out_filename = f"metrics_{emotion_name}.csv"
        df.to_csv(os.path.join(PROC_DIR, out_filename), index=False)
        print(f"Saved to {PROC_DIR}/{out_filename}")

print("\nALL VIDEOS PROCESSED! Pipeline Phase 1 complete")