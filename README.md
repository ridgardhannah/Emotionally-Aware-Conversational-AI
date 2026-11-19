# Emotionally Intelligent Conversational AI

**Author:** Hannah Ridgard  
**BSc (Hons) Software Engineering — Final Year Project**  
**Demonstration Video:** [Watch System Walkthrough on YouTube](https://youtu.be/xdQGVykLROI)

---

## 1. Project Summary

This repository contains the source code for a three-layer multimodal Affective AI pipeline designed to bridge the "empathy gap" in text-only conversational agents. By combining real-time computer vision, temporal physiological smoothing, and a locally hosted Large Language Model (LLM), the system evaluates non-verbal facial cues alongside typed user input to generate grounded, context-aware coaching responses.

### Key Architectural & Research Contributions

- **Perceptual Layer (Custom ResNet-18 CNN):** Modified a pre-trained ResNet-18 architecture with a custom 12-output sigmoid head to detect simultaneous activations across 12 target Facial Action Units (AUs) from the BP4D-Spontaneous dataset.
- **Masked Binary Cross-Entropy (BCE) Loss:** Engineered a custom dynamic loss-masking function to neutralise missing or occluded frame annotations (coded as `9` in BP4D) during backpropagation without discarding entire frames, achieving a final training loss of **0.0699** over 15 epochs.
- **Spatial Cropping & Pre-Processing:** Implemented geometric frame bisection prior to inference to isolate the target subject and prevent multi-face data contamination in split-screen footage.
- **Inference Layer & Evolutionary Calibration:** Built a temporal smoothing buffer (`collections.deque`) paired with a $(1+1)$ Evolutionary Strategy (ES) search to mathematically optimise AU activation thresholds (`thresholds.json`) and penalise false positives.
- **Privacy-First Local Deployment:** Executes end-to-end on consumer hardware using PyTorch, MediaPipe, Streamlit, and Meta's `llama3.2:1b` served locally via Ollama, ensuring biometric telemetry never leaves the user's device.

---

## 2. Code Structure

The codebase is modularly structured around the CRISP-DM lifecycle, progressing from dataset preparation and model training to threshold calibration and interactive deployment.

### Core Application & Data Pipeline

- `01_data_extractor.py`: Extracts frame-by-frame AU activation probabilities (ResNet-18) and ocular gaze vectors (MediaPipe Face Mesh) from raw `.mp4` footage into timestamped telemetry CSVs.
- `02_calibration.py`: Runs an Evolutionary Strategy optimisation loop over extracted telemetry and verified ground-truth windows to generate `thresholds.json`.
- `03new_app.py`: Interactive Streamlit dashboard orchestrating live video frame inference, real-time AU time-series plotting, radial cognitive state profiling, and local Ollama LLM coaching responses.
- `thresholds.json`: Evolved AU activation thresholds produced by the calibration stage.
- `requirements.txt`: Python package dependencies required to run the pipeline.

### Training, Dataset & Validation Utilities

- `train_resnet18.py`: Trains the modified ResNet-18 model using stochastic data augmentation and the custom Masked BCE loss function, generating `best_vision_model_EXPANDED_FINAL.pth` in the project root directory upon completion.
- `build_expanded_dataset.py`: Consolidates and aligns raw BP4D video frames with frame-level FACS annotations.
- `extract_specific_ranges.py`: Utility script for isolating targeted timestamp windows during ground-truth validation.
- `test_dataloader.py`: Fault-tolerant PyTorch `Dataset` wrapper with recursive exception handling for corrupted or truncated image files.
- `test_model_output.py`: Architectural verification test confirming that the modified ResNet-18 checkpoint outputs a strict 12-dimensional tensor.
- `subset_training_data.csv`: Sanitised training metadata mapping frames to target AU labels.
- `resnet18_training_curve.csv`: Recorded training loss metrics across 15 epochs.

---

## 3. Setup & Execution

### Prerequisites

- Python 3.10+ (Anaconda or virtual environment recommended)
- [Ollama](https://ollama.com/) installed locally with `llama3.2:1b` pulled

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate Model Weights

Due to repository size constraints and dataset licensing rules, pre-trained `.pth` weights and raw BP4D training frames are not bundled in this repository. Running the training script with the BP4D-Spontaneous dataset (obtained from the dataset creators) will train the vision backbone and generate `best_vision_model_EXPANDED_FINAL.pth` in the project root directory:

```bash
python train_resnet18.py
```

### Step 3: Start the Local LLM Server

Open a terminal window and start the local `llama3.2:1b` model via Ollama:

```bash
ollama run llama3.2:1b
```

### Step 4: Prepare Sample Videos & Launch the Dashboard

Create a `raw_videos/` folder in the project root containing `.mp4` test clips (named `mini_subject a.mp4` through `mini_subject d.mp4` to match the dashboard selector), then launch the Streamlit interface:

```bash
streamlit run 03new_app.py
```
