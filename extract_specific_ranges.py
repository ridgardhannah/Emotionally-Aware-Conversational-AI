import cv2
import os

#configs stuff
input_video = "test_video.mp4"
output_folder = "labelling_data"
FPS = 25  # Newsnight is usually 25fps

#define ranges. mins and secs
#format is (Start_Min, Start_Sec, End_Min, End_Sec, "Label_Hint")
#"Label_Hint" adds a word to the filename to hint what to look for in frame
ranges = [
    (0, 5,  0, 7,  "polite_smile"), 
    (0, 22, 0, 25, "neutral"),  #baseline  
    (0, 33, 0, 34, "frown"),  
    (0, 35, 0, 37, "raised_eyebrows_denial"),  
    (1, 10, 1, 11, "thinking_frown"),  
    (1, 28, 1, 29, "frown"),  
    (2, 19, 2, 20, "denial_sad")
]


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(input_video)

for start_m, start_s, end_m, end_s, label in ranges:
    #convert time to frame numbers
    start_frame = int((start_m * 60 + start_s) * FPS)
    end_frame = int((end_m * 60 + end_s) * FPS)
    
    print(f"Extracting '{label}' from {start_m}:{start_s} to {end_m}:{end_s} (Frames {start_frame}-{end_frame})...")
    
    #jump to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret: break
        
        #save every 5th frame to avoid dupes
        if (current_frame - start_frame) % 5 == 0:
            filename = f"{output_folder}/{current_frame}_frame_{label}.jpg"
            cv2.imwrite(filename, frame)
            print(f"  -> Saved {filename}")
            
        current_frame += 1

cap.release()
print("Done. check folder.")