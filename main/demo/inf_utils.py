import torch
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch.nn.functional as F
import json
import torchvision.transforms as T
from data_loader.data_loaders import evenly_sample_frames

def split_coordinates(input_list):
    x_coordinates = input_list[::3]
    y_coordinates = input_list[1::3]
    z_coordinates = input_list[2::3]

    return x_coordinates, y_coordinates, z_coordinates

def merge_coordinates(x_values, y_values, z_values):
    merged_list = [coord for triplet in zip(x_values, y_values, z_values) for coord in triplet]

    return merged_list


def normalize(coordinates):
    c_array = np.array(coordinates)

    # Calculate mean and covariance
    mean_value = np.mean(c_array)
    std_dev = np.std(c_array)

    # Normalize using mean and covariance
    normalized_coordinates = (c_array - mean_value) / std_dev

    return normalized_coordinates


def reshape_keypoints(keypoints_data):
    x, y, z = split_coordinates(keypoints_data)
    normalized_data = merge_coordinates(normalize(x), normalize(y), normalize(z))

    reshaped_keypoints = torch.tensor(normalized_data, dtype= torch.float32).reshape(-1, 3)

    return reshaped_keypoints


def processKeypoints(hand_landmarks):
    landmarks_list_per_hand = []
    for landmark in hand_landmarks.landmark:
        landmarks_list_per_hand.append(landmark.x)
        landmarks_list_per_hand.append(landmark.y)
        landmarks_list_per_hand.append(landmark.z)

    return landmarks_list_per_hand

def dict2list(landmarks_dict):  
    final_keypoints = []
    left_landmarks = []
    right_landmarks = []

    for frame in landmarks_dict:
        left_landmarks = reshape_keypoints(frame['Left'])
        right_landmarks = reshape_keypoints(frame['Right'])

        all_keypoints = torch.cat((left_landmarks, right_landmarks), dim=0)
        all_keypoints = all_keypoints.flatten()

        final_keypoints.append(all_keypoints)

    keypoints_data = torch.stack(final_keypoints)

    return keypoints_data

def init_hands():    
    with open('dummy_keypoint.json', 'r') as file:
        dummy_data = json.load(file)

    hand_right_dummy = dummy_data['hand_right_dummy']
    hand_left_dummy = dummy_data['hand_left_dummy']

    prev_dict = {}
    prev_dict['Right'] = hand_right_dummy
    prev_dict['Left'] = hand_left_dummy

    return prev_dict

def calculate_displacements(landmarks_dict, prev_dict, keypoints_queue):
    prev_dict = keypoints_queue.get() 

    # calc dist between correcsponding hand keypoints 
    displacements = []
    # average_displacement = 0
    for orientation, kps in landmarks_dict.items(): 
        if prev_dict[orientation] is None or kps is None: 
            continue 
            
        kps = np.delete(np.array(kps).reshape(-1, 3), 2, axis=1)
        prev = np.delete(np.array(prev_dict[orientation]).reshape(-1, 3), 2, axis=1)

        # get displacements between corresponding orientations 
        disp_by_ori = np.linalg.norm(kps - prev, axis=1)
        displacements.append(disp_by_ori)
    
    if len(displacements) > 0:
        displacements = np.concatenate(displacements, axis=0)
        total_disp = np.sum(displacements)
    else:
        total_disp = 0
    
    return total_disp


def clip2tensor(clip: list):

    frame_count = len(clip)
    sampled_indices = evenly_sample_frames(frame_count, 64, 0)
    sampled_frames = [clip[i] for i in sampled_indices]
    
    clip = torch.stack(sampled_frames, dim=0).transpose(0,1)
    clip = clip.unsqueeze(0)
    
    return clip

def process_clip(clip, device, input_type):
    if input_type == "RGB":
        clip_tensor = clip2tensor.to(device)
    elif input_type == "keypoint":
        clip_tensor = dict2list(clip).unsqueeze(0).to(device)

    return clip_tensor

def prepare_transforms(height):
    transforms = T.Compose([
            T.CenterCrop(height),  # Crop a square from the center
            T.Resize((224, 224)),           # Resize to 224x224
            T.ToTensor()
        ])
    return transforms

def get_model_outputs(model, clip_tensor):
    outputs = F.softmax(model(clip_tensor))
    return outputs

def process_outputs(lookup, outputs):
    indices = torch.argmax(outputs).item()
    word = lookup.iloc[indices]['word']
    
    values, indices = torch.topk(outputs, k=5, dim=-1)
    words = [lookup.iloc[idx]['word'] for idx in indices.tolist()[0]]
    
    return word, words, values, indices

def displayResult(word, image, y=210): 
    fontpath = "fonts/AppleGothic.ttf"
    font = ImageFont.truetype(fontpath, 50)
    b,g,r,a = 0,255,0,0
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((120, y),  word, font=font, fill=(b,g,r,a))
    img = np.array(img_pil)
    return img 