from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import json
import torch
import torch.nn.functional as F
from PIL import Image

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class KeyPointsDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = KeyPointDataset(path=data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class VideoDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = VideoDataset(path=data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class VideoDataset(Dataset):
    def __init__(self, path, mode='train', transforms=None, **kwargs):
        self.path = path
        self.mode = mode

        self.num_samples = 32
        self.interval = 0
        
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
            final_video_keys = self._get_splits(all_video_keys)

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

        sampled_indices = evenly_sample_frames(frame_count, self.num_samples)
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



    def _get_splits(self, video_keys):
        # Default split logic. Redefine the function if needed
        if self.mode == 'train':
            video_keys = video_keys[:int(len(video_keys)*0.8)]
        elif self.mode == 'val':
            video_keys = video_keys[int(len(video_keys)*0.8):int(len(video_keys)*0.9)]
        elif self.mode == 'test':
            video_keys = video_keys[int(len(video_keys)*0.9):]

        return video_keys

class KeyPointDataset(Dataset):
    def __init__(self, path, mode='train', transforms=None, **kwargs):
        self.path = path
        self.mode = mode

        self.num_samples = 128
        
        if transforms is None:
            self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
            ])
        else:
            self.transforms = transforms
        
        self.videos = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for vocab in os.listdir(self.path):
            label = int(vocab)
            for collector in os.listdir(os.path.join(self.path, vocab)):
                for angle in os.listdir(os.path.join(self.path, vocab, collector)):
                    vocab_datas = os.path.join(self.path, vocab, collector, angle)
                    self.videos.append(vocab_datas)
                    self.labels += [label for _ in range(len(vocab_datas))]

    def __len__(self):
        return len(self.videos)    

    def __getitem__(self, index):
        video_path = self.videos[index]
        frame_keys = sorted(os.listdir(video_path))
        frame_count = len(frame_keys)
        sampled_indices = evenly_sample_frames(frame_count, self.num_samples)
        sampled_keys = [frame_keys[idx] for idx in sampled_indices]

        keypoints = []

        for frame_key in sampled_keys:
            json_file = open(os.path.join(video_path, frame_key))
            data = json.load(json_file)
            json_file.close()
            
            # 4 (x,y,z, confidence) * 70 keypoints => [3, 70]
            face_keypoints = reshape_keypoints(data["people"]["face_keypoints_3d"])
            # 4 * 25 keypoints => [3, 25]
            pose_keypoints = reshape_keypoints(data["people"]["pose_keypoints_3d"])
            # 4 * 21 keypoints => [3, 21]
            hand_right_keypoints = reshape_keypoints(data["people"]["hand_right_keypoints_3d"])
            # 4 * 21 keypoints => [3, 21]
            hand_left_keypoints = reshape_keypoints(data["people"]["hand_left_keypoints_3d"])

            all_keypoints = torch.cat((face_keypoints, pose_keypoints, hand_right_keypoints, hand_left_keypoints), dim=0)
            keypoints.append(all_keypoints)
        
        # 128, 137, 3
        keypoints_data = torch.stack(keypoints, dim=0).transpose(0,2)
        keypoints_data = F.interpolate(keypoints_data.unsqueeze(0), size=(224,224)).squeeze(0)

        data = {'src': video_path, 'keypoints': keypoints_data, 'label': self.labels[index]}

        return keypoints_data, self.labels[index]

def evenly_sample_frames(total_frames, sample_count):
    frames_to_sample = min(total_frames, sample_count)
    step = total_frames // frames_to_sample
    sample_indices = [i * step for i in range(frames_to_sample)]
    return sample_indices

def reshape_keypoints(keypoints_data):
    del keypoints_data[4-1::4] # remove every 4th (confidence) value
    reshaped_keypoints = torch.tensor(keypoints_data).reshape(-1, 3)

    return reshaped_keypoints

if __name__ == '__main__':
    ds = VideoDataset()

    for d in ds:
        print(d['frame'].shape)