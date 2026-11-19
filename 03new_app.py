# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests  
import plotly.graph_objects as go

#configure web page
st.set_page_config(page_title="Emotionally Intelligent AI", layout="wide")

#create session state so streamlit remembers whats been clicked, analysied etc
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'selected_video' not in st.session_state:
    st.session_state.selected_video = "Subject A"
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None

#CSS Styling
#injected to override default streamlit appearance. colour buttons, pulsing process block, finds hidden markers etc
st.markdown("""
<style>
/* secondary buttons are turquoise (vid menu) */
div[data-testid="stButton"] button[kind="secondary"] {
    background-color: #06b6d4 !important;
    color: #000000 !important;
    font-weight: 900 !important;
    font-size: 22px !important;
    height: 4em !important;
    border: 4px solid #0891b2 !important;
    border-radius: 12px !important;
}

/* primary button is green for start button only */
div[data-testid="stButton"] button[kind="primary"] {
    background-color: #22c55e !important;
    color: #000000 !important;
    font-weight: 900 !important;
    font-size: 24px !important;
    height: 3.5em !important;
    border: 5px solid #166534 !important;
    border-radius: 12px !important;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.4) !important;
}

.active-subject-btn div[data-testid="stButton"] button {
    background-color: #334155 !important;
    color: #ffffff !important;
    border: 4px solid #000000 !important;
}
            
/*pulsing processing bar*/
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.6; }
    100% { opacity: 1; }
}

.processing-bar {
    background-color: #06b6d4 !important; /* turq like menu buttons */
    color: #000000 !important;
    padding: 15px !important;
    border-radius: 10px !important;
    border: 3px solid #0891b2 !important;
    font-weight: 900 !important;
    font-size: 18px !important;
    text-align: center !important;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3) !important;
    animation: pulse 1.5s infinite !important;
}

/*area for export buttons*/
div[data-testid="stDownloadButton"] button {
    font-weight: 900 !important;
    font-size: 16px !important;
    height: 3em !important;
    border-radius: 10px !important;
    width: 100% !important;
    color: #ffffff !important;
}

/* orange: find the container with CSV marker, then style the button in next container */
div.element-container:has(.csv-marker) + div.element-container button {
    background-color: #ff8c00 !important;
    border: 3px solid #cc7000 !important;
}

/* yellow: find container with the LOG marker then style the button in next container */
div.element-container:has(.log-marker) + div.element-container button {
    background-color: #ffdf00 !important;
    color: #000000 !important;
    border: 3px solid #ccb200 !important;
}
</style>
""", unsafe_allow_html=True)
#its safe so render it as html and css to change the colours etc

#map raw data to readable labels/descriptions
au_list = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU14", "AU15", "AU17", "AU23", "AU24"] 
au_desc = {
    "AU1": "Inner Brow Raise", "AU2": "Outer Brow Raise", "AU4": "Brow Lowerer",
    "AU6": "Cheek Raiser", "AU7": "Lid Tightener", "AU10": "Upper Lip Raiser",
    "AU12": "Lip Corner Pull", "AU14": "Dimpler", "AU15": "Lip Depressor",
    "AU17": "Chin Raiser", "AU23": "Lip Tightener", "AU24": "Lip Pressor"
}

#AI engine
#tells streamlit to load the heavy ai model into ram one time, when app starts. not everytime a button is clicked
@st.cache_resource
def load_assets():
    # asks motherboard in the nvidia, gpu/cuda is available or aapple silicon/mps.
    #if not, uses standard cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    #build empty resnet18 model, chop off default final layer
    #replace with custom final layer to output 12 aus
    model = models.resnet18(weights=None)

    #this is the custom layer. gives 12 outputs instead of 1000. 
    #randomly turn of half the neurons so it cant memorize the training data/overfitting
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 12))
    
    ##reload my trained weights into device detected, gpu/cpu
    model.load_state_dict(torch.load("best_vision_model_EXPANDED_FINAL.pth", map_location=device, weights_only=True))
    
    # move model to device
    model = model.to(device)
    model.eval()

    #resnet needs images to be 224x224 pixels. 
    #grab whatever frame comes out the video, crush it down to that size then convert it into the pytorch tensor/matrix format the gpu/cpu to do the calculatings
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    return model, transform, device # return the device so the loop knows where to send data

#header
#draw title and box for user to type in
st.title("Emotionally Intelligent Conversational AI")
goal_context = st.text_area(
    "What goals or tasks are you trying to manage? Maybe I can help! \n DISCLAIMER: This platform is in no way intended to act as a replacemement for medical help, nor is it a clinical diagnostic tool. If you feel in need of further support, please contact a medical professional. ",
    placeholder="Describe your current objective...",
    value="Finish my Final Year Project"
)
st.markdown("---")

#subject selector
st.markdown("### SELECT SUBJECT")
#create 4 columns, loop through subject list to draw buttons.
subjects = ["Subject A", "Subject B", "Subject C", "Subject D"]
v_cols = st.columns(4)


#button click logic
for i, sub in enumerate(subjects):
    #if selected, save subject to memory as sub
    is_active = st.session_state.selected_video == sub
    with v_cols[i]:
        #if a subject button is clicked
        if is_active:
            #clean slate for currently clicked button.
            st.markdown('<div class="active-subject-btn">', unsafe_allow_html=True)
        clicked = st.button(sub, key=f"subj_{sub}", use_container_width=True, type="secondary")
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)
        if clicked:
            #new choice saved to mem. reset analysis status to false so radial is hiden etc
            st.session_state.selected_video = sub
            st.session_state.analysis_done = False
            st.session_state.last_frame = None
            #force webpage to refresh to update ui
            st.rerun()

st.markdown("---")

#dashboard
#sets up main layout. vid gets 2 parts of screen, graph gets 2, contrl panel gets 1
col_v, col_g, col_rhs = st.columns([2, 2, 1], gap="large")

#vid column
with col_v:
    #create empty placeholder boxes to dynamically inject video, buttons, charts etc
    video_placeholder = st.empty()
    if st.session_state.last_frame is not None:
        video_placeholder.image(st.session_state.last_frame, use_container_width=True)
    else:
        video_placeholder.info(f"Selected Video: {st.session_state.selected_video}")
    verdict_box_placeholder = st.empty()

#au graph column
with col_g:
    chart_placeholder = st.empty()
    chart_placeholder.line_chart(pd.DataFrame(np.zeros((50, 12)), columns=au_list))
    st.markdown("#### BIOMETRIC KEY")
    k_rows = [st.columns(3) for _ in range(4)]
    for idx, au in enumerate(au_list):
        row, col_idx = idx // 3, idx % 3
        k_rows[row][col_idx].caption(f"**{au}**: {au_desc[au]}")

#control panel column
with col_rhs:
    st.markdown("### CONTROL PANEL")
    analyse_btn = st.button("▶ START ANALYTICS", key="start_btn", use_container_width=True, type="primary")

    st.markdown("<br>", unsafe_allow_html=True)
    frame_metric = st.empty()
    status_indicator = st.empty()

    st.markdown("---")
    st.markdown("### EXPORT DATA")
    #space for 2 buttons
    e_col1, e_col2 = st.columns(2)


    #checks if data has been crunched yet.
    #if yes, export data buttons are clickable
    if st.session_state.analysis_done:
        with e_col1:
            st.markdown('<span class="csv-marker" style="display:none;"></span>', unsafe_allow_html=True)
            st.download_button("⬇ CSV", data=st.session_state.final_csv,
                               file_name=f"data_{st.session_state.selected_video}.csv",
                               use_container_width=True, key="dl_csv_final")
        with e_col2:
            st.markdown('<span class="log-marker" style="display:none;"></span>', unsafe_allow_html=True)
            st.download_button("⬇ LOG", data=st.session_state.audit_log,
                               file_name=f"audit_{st.session_state.selected_video}.txt",
                               use_container_width=True, key="dl_log_final")
     #if no disable them       
    else:
        with e_col1:
            st.markdown('<span class="csv-marker" style="display:none;"></span>', unsafe_allow_html=True)
            st.download_button("⬇ CSV", data="", file_name="pending.csv",
                               use_container_width=True, key="dl_csv", disabled=True)
        with e_col2:
            st.markdown('<span class="log-marker" style="display:none;"></span>', unsafe_allow_html=True)
            st.download_button("⬇ LOG", data="", file_name="pending.txt",
                               use_container_width=True, key="dl_log", disabled=True)

#post-analysis
st.markdown("---")
#if done look at final emotional state.
if st.session_state.analysis_done:
    
    #assign colour depending on emotional state
    if st.session_state.mood == "Positive/High Momentum":
        bg_color = "#22c55e" # Green
    elif st.session_state.mood == "Fatigued/Cognitively Depleted":
        bg_color = "#64748b" # Slate Grey
    elif st.session_state.mood == "Distressed/Overwhelmed":
        bg_color = "#f97316" # Warm Orange 
    else:
        bg_color = "#3b82f6" # Steady Blue

    #large state banner. give it colour and emotional state
    verdict_box_placeholder.markdown(
        f'<div style="background-color:{bg_color};color:white;padding:20px;border-radius:15px;'
        f'text-align:center;border:3px solid white;box-shadow:0px 5px 15px rgba(0,0,0,0.3);">'
        f'<h1>STATE: {st.session_state.mood}</h1></div>', unsafe_allow_html=True)

    #draw ai coach response box full width. add generated text
    st.markdown("#### AI TACTICAL COACH")
    if 'coach_text' in st.session_state:
        st.success(st.session_state.coach_text)
        
    st.markdown("<br>", unsafe_allow_html=True) 

    #split row for radial graph and explanations
    bot_cols = st.columns([1.5, 2])
    
    #radial graph
    with bot_cols[0]:
        st.markdown("#### AFFECTIVE FINGERPRINT")
        #in the radial graph:
        fig = go.Figure(go.Scatterpolar(
            #take 4 final peak numbers, j, s, t and f. assign to labels 
            r=[st.session_state.j, st.session_state.s, st.session_state.t, st.session_state.f],
            #labels for radial graph
            theta=['Smile Activation', 'Brow Compression', 'Lower Face Tension', 'Ocular Engagement'],
            fill='toself', line_color='#22c55e', fillcolor='rgba(34,197,94,0.4)'))
        #fill and show
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                          showlegend=False, height=450, margin=dict(l=40,r=40,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    #radial graph explanation in laymans terms.
    #if else block to work out which to show
    with bot_cols[1]:
        st.markdown("#### WHAT DOES THIS MEAN?")
        if st.session_state.mood == "Positive/High Momentum":
            st.info("**Genuine Engagement Detected:** Your graph might show high **Ocular Engagement** alongside **Smile Activation**. This doesn't mean you are just staring hard! It means the muscles around your eyes are crinkling (the genuine, Duchenne Smile marker)! This proves your smile is biologically authentic and not forced. You are in a great flow state!")
        
        elif st.session_state.mood == "Fatigued/Cognitively Depleted":
            st.warning("**Mental Drain Detected:** Your graph is showing sustained **Brow Compression**. Unlike sharp spikes of stress, this heavy, drooping profile usually means your brain is working overtime and your cognitive battery is running low. It's time to step back.")
        
        elif st.session_state.mood == "Distressed/Overwhelmed":
            st.warning("**Friction Detected:** Your graph shows high **Brow Compression** as your eyebrows pull together. You may also see spikes in **Smile Activation**—don't worry, the system isn't confused! Humans use the exact same cheek and lip muscles to grimace or sob as they do to smile. Your face is physically reacting to high cognitive load.")
        
        else:
            st.success("**Baseline State:** Your facial muscles are relaxed, with low **Brow Compression** and minimal **Lower Face Tension**. You are in a steady, productive baseline state with no extreme emotional spikes pulling your attention away.")

#when the user clicks start button, get the loaded model and hardware device
if analyse_btn:
    model, transform, device = load_assets()

    #work out video to play based on the saved subject from button click
    #works by turning button name to lower case and hunting through vid folder for mp4 file
    video_choice = st.session_state.selected_video
    #select correct vid and 
    target = f"mini_{video_choice.lower()}"

    #scan different file path options until vid is found. stop when it finds it
    found_file = next(
        (os.path.join(p, f) for p in ["raw_videos", "./raw_videos", "../raw_videos"]
         if os.path.exists(p) for f in os.listdir(p) if f.lower().startswith(target)), None)

    #handle not finding the vid
    if not found_file:
        st.error(f"Target video missing for {target}.")
        st.stop()

    #open vid.
    cap = cv2.VideoCapture(found_file)
    history, frame_count, last_rgb = [], 0, None
    
    status_indicator.markdown('<div class="processing-bar">Status: Processing Biometric Data...</div>', unsafe_allow_html=True)

    #loop through and read it frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        #skip some frames to make performance better on cpu
        if frame_count % 10 == 0:
            #convert frame to standard rgb from the weird opencv bgr
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_rgb = rgb
            
            video_placeholder.image(rgb, use_container_width=True)
            frame_metric.metric("Frames Analysed", frame_count)
            
            #core ai inference section.
            #convert image into mathematical tensor and ship it to cpu/gpu
            tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            #push it through resnet model.
            with torch.no_grad():
                #outputted number will be raw and messy. sigmoid functions squash numbers to fall perfectly between 0 and 1 for confidence
                #torch.sigmoid squashes output so all 12 nums fall between 0.0 and 1.0 to be %ages
                #basically strips away the pytorch ness to get a list of normal numbers
                probs = torch.sigmoid(model(tensor))[0].tolist()

            #zip joins the 12 au labels with the 12 probabilities into a dictionary, saved to history  
            history.append(dict(zip(au_list, probs)))

            #updates live graph. uses -50: so it only draws the most recent 50 frames
            #stops browser crashing
            chart_placeholder.line_chart(pd.DataFrame(history[-50:]))

    cap.release()
    status_indicator.empty()
    #freeze last frame so vid space doesnt go blank
    if last_rgb is not None:
        st.session_state.last_frame = last_rgb

    #use defaults if thresholds file wont open or load
    try:
        with open('thresholds.json', 'r') as f: evolved = json.load(f)
    except:
        evolved = {"T_duchenne": 0.56, "T_brow_stress": 0.10, "T_brow_fatigue": 0.28}

    final_df = pd.DataFrame(history)
    st.session_state.final_csv = final_df.to_csv().encode('utf-8')
    peaks = final_df.max() if not final_df.empty else pd.Series(np.zeros(12), index=au_list)

    #average au6 and 12 to get smile cluster/joy
    j = (peaks.get("AU6", 0) + peaks.get("AU12", 0)) / 2
    
    #isolate au4 for stress/brow lowerer
    s = peaks.get("AU4", 0)
    
    #averages au10, upper lip raiser and au17, chin raiser. 
    #when tense, we grimace and tighten chin/raise upper lip a little like a sneer
    t = (peaks.get("AU10", 0) + peaks.get("AU17", 0)) / 2
    
    #isolates au7 lid tightener for engagement/squinting
    f = peaks.get("AU7", 0)

    #threshold logic using evolutionary strategy results
    #categorises emotional state
    #positive. does j beat 0.56 threshold AND is s relaxed below 0.1?
    if j > evolved.get("T_duchenne", 0.56) and s < evolved.get("T_brow_stress", 0.10):
        mood, reason = "Positive/High Momentum", "Smile cluster met thresholds with relaxed eyebrows."
    
    #fatigud. if not positive is the brow heavily compressed more than 0.28?
    elif s >= evolved.get("T_brow_fatigue", 0.28):
        mood, reason = "Fatigued/Cognitively Depleted", "High brow activation (sagging profile) detected."
    
    #distressed. if not positive or fatigued is the brow a little compressed, more than 0.1?
    elif s >= evolved.get("T_brow_stress", 0.10):
        mood, reason = "Distressed/Overwhelmed", "Moderate brow furrowing detected within stress limits."
    
    #if none of those rules are triggered it must be baseline neutral steady emotion
    else:
        mood, reason = "Neutral/Steady", "Affective markers below trigger thresholds."

    audit_log = f"""AFFECTIVE ARCHITECT - DECISION AUDIT
Subject: {video_choice}
-------------------------------------------------
1. PEAK BIOMETRIC DATA:
/* take the raw long number and round it to 4 decimal places, like 0.xxxx. that way text log is readable*/
- Joy Cluster: {j:.4f}
- Stress/Fatigue (Brow): {s:.4f}
- Face Tension: {t:.4f}
- Ocular Focus: {f:.4f}

2. SYSTEM REASONING:
- Triggered Logic: {reason}
- Comparison Boundary (Stress): {evolved.get('T_brow_stress', 0.10):.4f}
- Comparison Boundary (Fatigue): {evolved.get('T_brow_fatigue', 0.28):.4f}

3. FINAL RESULT:
- State: {mood}
-------------------------------------------------"""

    #all local variables like j, s, mood etc are deleted by python
    #pinning them to session state saves in browsers memory
    #when page reloads to show final results the radial chart and logs can access the final data
    st.session_state.j, st.session_state.s = j, s
    st.session_state.t, st.session_state.f = t, f
    st.session_state.mood = mood
    st.session_state.audit_log = audit_log.encode('utf-8')

    #local ollama ref
    #take custom user text input (goal context) and ai final mood calculation and puts them together
    try:
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": "llama3.2:1b",
            "messages": [
                {"role": "system", "content": "You are a supportive productivity and wellness coach. The user is completing their own work. Your job is ONLY to provide 3 brief time-management or stress-relief tips based on their biometric state. Do not write their project for them. Keep it brief, professional, and encouraging."},
                {"role": "user", "content": f"The user is currently working on: {goal_context}. Their current cognitive state is: {mood} ({reason}). Give 3 quick tips to help them manage their focus and wellbeing right now."}
            ],
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            st.session_state.coach_text = response.json()['message']['content']
        else:
            st.session_state.coach_text = "Coach unavailable. Ollama returned an error."
            
    except requests.exceptions.RequestException:
        st.session_state.coach_text = "Coach unavailable. Please ensure your local Ollama server is running!"

    #mark coaching process as complete. relads ui so dashboard populates with data, charts and advice
    st.session_state.analysis_done = True
    st.rerun()