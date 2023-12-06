# KSLR (Korean Sign Language Recognition)
</br>
<h3 align="center"> :wave: The 3rd YAICON project - Multiframe-based Sign Language Translator </h3>
</br>
<p align="center"><img src="https://github.com/cygbbhx/kslr/blob/main/img/demo.gif" width="50%" height="50%"></p>
</br>

---
## Introduction
This project focuses on video classification at the word level, specifically targeting Korean Sign Language (KSL). By taking word-level sign language videos as input, our system classifies the signs into distinct categories. The model is designed to process approximately 64 frames of video, providing classification for 60 different sign language words. Explore our repository to delve into the implementation and contribute to the development of this meaningful initiative.

## Team Members
<b>
:point_up: Sohyun Yoo (YAI 12th) - Video data preprocessing / Model Experiments / Lead </br>
:v: Meedeum Cho (YAI 11th) - Segmentation / Real-time Demo </br>
:ok_hand: Hyunjin Park (YAI 12th) - Keypoint data analysis & preprocessing </br>
:facepunch: Jeongmin Seo (YAI 12th) - Model Experiments / Data collecting </br>
:raised_hand_with_fingers_splayed: Hyemi Yoo (YAI 12th) - Model Experiments / Real-time Demo </br>
</b>

## Dataset
We used the following dataset for our Korean Sign Language Translation project:
- [AIhub Sign Language Video Dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=103)
- After downloading the dataset, you will have to convert videos into images of frames. We have utilized the generation code `./utils/generate_data.sh` from [here](https://github.com/pranoyr/cnn-lstm/). Environment setup for the preprocessing is also included in the repository's README. 
- For the AIhub dataset, we also provide you a code for easier preprocessing as below:
  - `preprocessing/rearrange_videos.py`: extracts only the videos that are in `target_words.txt` from all zipfiles and rearranges them into directories of classes (words). This code will be useful if you want only a subset, rather than extracting all the zipfiles.
  
## Setup
To set up the project, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/cygbbhx/kslr.git
cd kslr
```
### 2. Install Dependencies
Create and activate a virtual environment using anaconda:
```bash
conda create --name kslr-env python=3.8
conda activate kslr-env
```

Install the required packages:
```bash
pip install -r requirements.txt
```


## Training the Model
To train the model, run the following command:
```bash
python train.py
```
TBU

### config
TBU

## Inference
### Model weights
TBA

### Inference with video inputs
TBU
```bash
python inference.py --input_file path/to/input/video.mp4
```

### Inference with Real-time Webcam (Demo)
To try the demo with real world input, run `demo.py` below main/demo directory. 
```
cd demo
python demo.py -w path_to_model_weight.pt
```
- You can adjust the arguments to change the input to be RGB (pixel values) or keypoints. In our implementation, the code automatically changes the input type according to the model choice.
