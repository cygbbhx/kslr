# KSLR (Korean Sign Language Recognition)
</br>
<h3 align="center"> :wave: The 3rd YAICON project - Multiframe-based Sign Language Translator </h3>
</br>
<p align="center"><img src="https://github.com/cygbbhx/kslr/blob/main/img/demo.gif" width="50%" height="50%"></p>
</br>

---


## Team Members
- [Member 1] - Role
- [Member 2](#) - Role
- [Member 3](#) - Role

## Dataset
We used the following dataset for our Korean Sign Language Translation project:
- [aihub Sign Language Video Dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=103)
  
## Setup
To set up the project, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/cygbbhx/kslr.git
cd kslr
```
### 2. Install Dependencies
Create and activate a virtual environment using [conda](https://docs.conda.io/projects/conda/en/latest/index.html):
```bash
conda create --name kslr-env python=3.8
conda activate kslr-env
```

Install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Data Preprocessing
Explain any necessary data preprocessing steps here.

## Training the Model
To train the model, run the following command:
```bash
python train.py
```

Explain any additional options or parameters that can be used during training.

## Inference
### Inference with video inputs
To perform inference using the trained model, use the following command:
```bash
python inference.py --input_file path/to/input/video.mp4
```
### Inference in real-time Webcam
Provide any additional instructions or options for the inference process.
```
