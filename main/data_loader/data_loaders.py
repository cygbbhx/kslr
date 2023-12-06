from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
import math
import numpy as np

class KeyPointsDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, **kwargs):
        self.dataset = KeyPointDataset(path=data_dir, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class VideoDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, **kwargs):
        self.dataset = VideoDataset(path=data_dir, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class VideoDataset(Dataset):
    def __init__(self, path, mode='train', transforms=None, **kwargs):
        self.path = path
        self.mode = mode

        self.num_samples = kwargs.get('frames')
        self.interval = kwargs.get('interval')
        
        self.sample_even = kwargs.get('sample_even')
        self.sampling = evenly_sample_frames if self.sample_even else uniform_sample_frames

        if transforms is None:
            self.transforms = T.Compose([
                T.CenterCrop(300),
                T.RandomApply([T.RandomAffine(degrees=5, translate=(0.05, 0.05))], p=0.5),  # Random rotation and shifting
                T.RandomResizedCrop(224, scale=(0.7, 1.3), ratio=(0.8, 1.2)),  # Random resizing and cropping Resize to 224x224
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
                T.ToTensor()
            ])
        else:
            self.transforms = transforms
        
        self.videos = []
        self.labels = []
        
        # load data after init
        self._load_data()

    def _load_data(self):
        vocab_dirs = sorted(os.listdir(self.path))

        for i, video_dir in enumerate(vocab_dirs):
            video_path = os.path.join(self.path, video_dir)
            assert os.path.exists(video_path), f"{video_dir} does not exist"

            all_video_keys = sorted(os.listdir(video_path))
            final_video_keys = all_video_keys #TBF

            video_dirs = [os.path.join(video_path, video_key) for video_key in final_video_keys]
            self.videos += video_dirs
            self.labels += [i for _ in range(len(final_video_keys))]

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        video_dir = self.videos[index]
        frame_keys = sorted(os.listdir(video_dir))
        frame_keys = [frame_key for frame_key in frame_keys if frame_key.endswith('.jpg')]
        frame_keys = trim_action(video_dir, frame_keys)
        frame_count = len(frame_keys)

        sampled_indices = self.sampling(frame_count, self.num_samples, self.interval)
        sampled_keys = [frame_keys[idx] for idx in sampled_indices]

        frames = []

        # Fix the randomness across the video for all frames
        state = torch.get_rng_state()   
        for frame_key in sampled_keys:
            frame = Image.open(os.path.join(video_dir, frame_key))
            torch.set_rng_state(state)
            frame = self.transforms(frame)
            frames.append(frame)
        #I3D
        frame_data = torch.stack(frames, dim=0).transpose(0,1)

        #CNNLSTM
        # frame_data = torch.stack(frames, dim=0)

        label = self.labels[index]
        print(video_dir, label)

        return frame_data, label


class KeyPointDataset(Dataset):
    def __init__(self, path, mode='train', transforms=None, **kwargs):
        self.path = path
        self.mode = mode

        self.num_samples = kwargs.get('frames')
        self.interval = kwargs.get('interval')

        self.sample_even = kwargs.get('sample_even')
        self.sampling = evenly_sample_frames if self.sample_even else uniform_sample_frames
        
        self.keypoint_types = kwargs.get('keypoints')
        self.framework = kwargs.get('framework')
        print(f"using keypoints: {self.keypoint_types}")

        self.format = kwargs.get('format')

        if transforms is None:
            aug_list = [T.ToTensor()]
            if self.format == 'image':
                aug_list.insert(T.Resize((224, 224)))

            self.transforms = T.Compose(aug_list)
        else:
            self.transforms = transforms
        
        self.videos = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        vocabs = sorted(os.listdir(self.path))
        for i, vocab in enumerate(vocabs):
            label = i
            if self.framework == 'mediapipe':
                for instance in os.listdir(os.path.join(self.path, vocab)):
                    vocab_datas = os.path.join(self.path, vocab, instance)
                    self.videos.append(vocab_datas)
                    self.labels.append(label)
            else: #openpose, original data
                for collector in os.listdir(os.path.join(self.path, vocab)):
                    for angle in os.listdir(os.path.join(self.path, vocab, collector)):
                        vocab_datas = os.path.join(self.path, vocab, collector, angle)
                        self.videos.append(vocab_datas)
                        self.labels.append(label)

    def __len__(self):
        return len(self.videos)    

    def __getitem__(self, index):
        video_path = self.videos[index]
        frame_keys = sorted(os.listdir(video_path))
        frame_keys = trim_action(video_path, frame_keys)
        
        frame_count = len(frame_keys)
        sampled_indices = self.sampling(frame_count, self.num_samples, self.interval)
        sampled_keys = [frame_keys[idx] for idx in sampled_indices]

        keypoints = []
        prev_key = None
        idx = 1

        for frame_key in sampled_keys:
            json_file = open(os.path.join(video_path, frame_key))
            data = json.load(json_file)
            json_file.close()

            total_keypoints = []

            for keypoint_type in self.keypoint_types:
                if self.framework == 'openpose':
                    type_keypoints = data["people"][f"{keypoint_type}_keypoints_2d"]
                    type_keypoints = reshape_keypoints(type_keypoints)
                else: # mediapipe json
                    assert f"{keypoint_type}_keypoints" in data.keys(), f"{keypoint_type}_keypoints not exist in {video_path}/{frame_key}"
                    type_keypoints = torch.tensor(data[f"{keypoint_type}_keypoints"])
                    type_keypoints = reshape_keypoints(type_keypoints)
                # averaged_keypoints = type_keypoints.mean(dim=1)
                total_keypoints.append(type_keypoints)

            all_keypoints = torch.cat((total_keypoints), dim=0)

            if self.format != 'image':
                all_keypoints = all_keypoints.flatten()
            if self.format == 'flatten':
                average_value = torch.tensor(all_keypoints).mean()
                all_keypoints = average_value

            keypoints.append(all_keypoints)

        keypoints_data = torch.stack(keypoints)  # Shape: [self.num_samples, 3 * num_keypoints]

        
        if self.format == 'flatten':
            keypoints_data = keypoints_data.view(1, self.num_samples)  # Reshape to [1, self.num_samples] to match LSTM input

        return keypoints_data, self.labels[index]

def trim_action(video_path, frame_keys):
    last_dir = video_path.split('/')[-1]
    vocab = video_path.split('/')[-2]
    morpheme_path = '../../../kslr_metadata/train_morphemes'

    # video data
    if len(last_dir) > 1:
        _, _, _, collector, angle = last_dir.split('_')
        collector = collector[4:] # REAL01 => 01

    json_filename = os.listdir(os.path.join(morpheme_path, vocab, collector, angle))[0]
    json_filepath = os.path.join(os.path.join(morpheme_path, vocab, collector, angle, json_filename))

    jsonfile = open(json_filepath)
    data = json.load(jsonfile)
    start = data["data"][0]["start"]
    end = data["data"][0]["end"]

    # print(f"start: {start} | end: {end}")

    start_frame = math.floor(start * 30)
    end_frame = math.ceil(end * 30)
    # print(f"start frame: {start_frame} | end_frame: {end_frame}")
    # print(f"originally {len(frame_keys)} frames")

    trimmed = frame_keys[start_frame: end_frame]

    # print(f"{len(trimmed)} frames")

    return trimmed
    

def evenly_sample_frames(total_frames, sample_count, interval):
    frames_to_sample = min(total_frames, sample_count)
    step = total_frames // frames_to_sample
    sample_indices = [i * step for i in range(frames_to_sample)]
    return sample_indices

def uniform_sample_frames(total_frames, sample_count, interval):
    if interval == 0:
        sample_indices = list(range(total_frames))[:sample_count]
    else:
        sample_indices = list(range(0, total_frames, interval))[:sample_count]
    
    while len(sample_indices) < sample_count:
        sample_indices.append(sample_indices[-1])

    return sample_indices

def reshape_keypoints(keypoints_data):
    # Removes every 4th or 3rd value. Uncomment according to your keypoint formats
    # del keypoints_data[4-1::4] 
    # del keypoints_data[2::3]

    # Divide every 1st value by 1920 and every 2nd value by 1080
    # for i in range(0, len(keypoints_data), 2):
    #     keypoints_data[i] /= 1920
    #     keypoints_data[i + 1] /= 1080

    x, y, z = split_coordinates(keypoints_data)
    normalized_data = merge_coordinates(normalize(x), normalize(y), normalize(z))

    reshaped_keypoints = torch.tensor(normalized_data).reshape(-1, 3)

    return reshaped_keypoints

if __name__ == '__main__':
    ds = VideoDataset()

    for d in ds:
        print(d['frame'].shape)

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