import pandas as pd
import json
import os

print("STARTING MULTI-AU FUSION CALIBRATION")

#verified manual timestamps from screen recording session
manual_timestamps = {
    "Angry":   {"start_sec": 2.0, "end_sec": 4.0},
    "Disgust": {"start_sec": 3.0, "end_sec": 6.0},
    "Happy":   {"start_sec": 6.0, "end_sec": 8.0},
    "Sad":     {"start_sec": 3.0, "end_sec": 5.0},
    "Tired":   {"start_sec": 3.0, "end_sec": 5.0},
    "Worried": {"start_sec": 7.0, "end_sec": 8.0}
}

PROC_DIR = "processed_data"

#these clusters let the logic distinguish between overlapping states
emotion_profiles = {
    "Angry":   {"file": "metrics_angry.csv",   "aus": ["AU4", "AU7", "AU23"]}, 
    "Disgust": {"file": "metrics_disgust.csv", "aus": ["AU10", "AU17"]},       
    "Happy":   {"file": "metrics_happy.csv",   "aus": ["AU6", "AU12"]},        
    "Sad":     {"file": "metrics_sad.csv",     "aus": ["AU1", "AU4", "AU15"]}, 
    "Tired":   {"file": "metrics_tired.csv",   "aus": ["AU4", "AU7"]},         
    "Worried": {"file": "metrics_worried.csv", "aus": ["AU1", "AU2", "AU4"]}   
}

master_profile = {}
all_buffers = []

for emotion, data in emotion_profiles.items():
    filename = data["file"]
    target_aus = data["aus"]
    file_path = os.path.join(PROC_DIR, filename)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        t_start = manual_timestamps[emotion]["start_sec"]
        t_end = manual_timestamps[emotion]["end_sec"]
        ground_truth_frames = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)].index.tolist()
        
        emotion_au_data = {}
        
        #run the ES sweep for every muscle in the cluster
        for au in target_aus:
            baseline = df[au].iloc[:15].mean()
            
            best_buffer = 0.05
            best_score = -999
            
            #bottom-up sweep to find the most sensitive threshold not including noise
            for gen in range(1, 31): 
                curr_buffer = gen * 0.02
                test_threshold = baseline + curr_buffer
                detected = df[df[au] > test_threshold].index.tolist()
                
                hits = len(set(detected).intersection(set(ground_truth_frames)))
                false_positives = len(detected) - hits
                score = (hits * 2.0) - (false_positives * 2.0)
                
                if score > best_score:
                    best_score = score
                    best_buffer = curr_buffer
                    
            emotion_au_data[au] = {
                "baseline_mean": round(baseline, 4),
                "es_optimized_buffer": round(best_buffer, 4)
            }
            all_buffers.append(best_buffer)
            
        #gett overall peak gaze for the emotion with the primary AU as the anchor
        window_df = df.loc[ground_truth_frames]
        if not window_df.empty:
            peak_idx = window_df[target_aus[0]].idxmax() 
            peak_gaze = df.loc[peak_idx, 'Gaze_Direction']
        else:
            peak_gaze = "FORWARD"
            
        master_profile[emotion] = {
            "aus": emotion_au_data,
            "typical_gaze": peak_gaze
        }
        print(f"[{emotion.upper()}] Calibrated Cluster: {target_aus}")
    else:
        print(f"could not find {filename}. Check processed_data folder")

if all_buffers:
    global_avg_buffer = sum(all_buffers) / len(all_buffers)
    final_data = {
        "profiles": master_profile,
        "global_avg_buffer": round(global_avg_buffer, 4)
    }
    with open('calibration_profiles.json', 'w') as f:
        json.dump(final_data, f, indent=4)
    print(f"\nDone. calibration_profiles.json has been updated with Multi-AU data")