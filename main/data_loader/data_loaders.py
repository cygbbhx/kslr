from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import json
import torch
import torch.nn.functional as F
from PIL import Image

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

        frame_data = torch.stack(frames, dim=0).transpose(0,1)
        label = self.labels[index]

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
        frame_count = len(frame_keys)
        sampled_indices = self.sampling(frame_count, self.num_samples, self.interval)
        sampled_keys = [frame_keys[idx] for idx in sampled_indices]

        keypoints = []
        prev_key = None
        idx = 1

        for frame_key in sampled_keys:
            try:
                json_file = open(os.path.join(video_path, frame_key))
                data = json.load(json_file)
                json_file.close()
            except Exception as e:
                print(f"{video_path}/{frame_key}", e)
                prev_key_file = open(os.path.join(video_path, prev_key))
                data = json.load(prev_key_file)
                with open(os.path.join(video_path, frame_key), 'w') as f:
                    json.dump(data, f, indent=4)
                prev_key_file.close()

            total_keypoints = []

            for keypoint_type in self.keypoint_types:
                if self.framework == 'openpose':
                    type_keypoints = data["people"][f"{keypoint_type}_keypoints_3d"]
                    type_keypoints = reshape_keypoints(type_keypoints)
                else: # mediapipe json
                    if f"{keypoint_type}_keypoints" not in data.keys():
                        while prev_key == None:
                            cand_file = open(os.path.join(video_path, frame_keys[idx]))
                            cand_data = json.load(cand_file)
                            cand_file.close()

                            if f"{keypoint_type}_keypoints" in cand_data.keys():
                                prev_key = frame_keys[idx]
                            else:
                                idx += 1

                        prev_key_file = open(os.path.join(video_path, prev_key))
                        data = json.load(prev_key_file)
                        with open(os.path.join(video_path, frame_key), 'w') as f:
                            json.dump(data, f, indent=4)
                        prev_key_file.close()
                    
                    assert f"{keypoint_type}_keypoints" in data.keys(), f"{keypoint_type}_keypoints not exist in {video_path}/{frame_key}"
                    type_keypoints = torch.tensor(data[f"{keypoint_type}_keypoints"])
                    prev_key = frame_key
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
    del keypoints_data[4-1::4] # remove every 4th (confidence) value
    reshaped_keypoints = torch.tensor(keypoints_data).reshape(-1, 3)

    return reshaped_keypoints

if __name__ == '__main__':
    ds = VideoDataset()

    for d in ds:
        print(d['frame'].shape)
