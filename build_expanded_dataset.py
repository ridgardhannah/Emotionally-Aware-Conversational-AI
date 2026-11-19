import os
import pandas as pd

# Define paths
bp4d_drive_path = r"G:\My Drive\BP4D"

#7 baseline subjects
baseline_subjects = ["F001", "F002", "F003", "F004", "M001", "M002", "M003"]

#include all tasks to introduce the new expressions while keeping 1, 2, and 7 to prevent forgetting
all_tasks = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"] 

expanded_data = []

print("Building Expanded Training CSV")

for subject in baseline_subjects:
    for task in all_tasks:
        images_dir = os.path.join(bp4d_drive_path, subject, task)
        csv_path = os.path.join(bp4d_drive_path, "AUCoding", "AU_OCC", f"{subject}_{task}.csv")
        
        #only process if both the image folder and csv exist
        if os.path.exists(images_dir) and os.path.exists(csv_path):
            print(f"Processing {subject} - {task}...")
            df = pd.read_csv(csv_path)
            
            for index, row in df.iterrows():
                frame_number = int(row['0'])
                image_name = f"{frame_number:04d}.jpg"
                image_path = os.path.join(images_dir, image_name)
                
                #if the image exists, add it to master list
                if os.path.exists(image_path):
                    row_data = {'ImagePath': image_path, 'Subject': subject, 'Task': task}
                    
                    #add all the AU columns
                    for col in df.columns[1:]:
                        row_data[col] = row[col]
                        
                    expanded_data.append(row_data)

#convert to dataframe and save
expanded_df = pd.DataFrame(expanded_data)
expanded_df.to_csv("expanded_training_data.csv", index=False)

print("\nSummary")
print(f"Done Expanded CSV created with {len(expanded_df)} total frames ready for training.")