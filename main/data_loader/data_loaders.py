from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import json
import torch
import torch.nn.functional as F

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
        self.dataset = KeyPointDataset()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class VideoDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = VideoDataset()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class VideoDataset(Dataset):
    def __init__(self, path='../../kslr_dataset/train', mode='train', transforms=None, **kwargs):
        self.path = path
        self.mode = mode

        self.num_samples = kwargs['num_samples']
        self.interval = kwargs['interval']
        
        if transforms is None:
            self.transforms = T.ToTensor()
        else:
            self.transforms = transforms
        
        self.videos = []
        self.labels = []
        
        self.mtype_index = []
        self.clips = []
        self.clip_src_idx = []
        
        # load data after init
        self._load_data()

    def _load_data(self):
        
        assert self.iter_path is not None, "video directories are not set"
        assert self.mtype is not None, "manipulation types are not set"

        for i, video_dir in enumerate(self.iter_path):
            assert os.path.exists(video_dir), f"{video_dir} does not exist"

            all_video_keys = sorted(os.listdir(video_dir))
            final_video_keys = self._get_splits(all_video_keys)

            video_dirs = [os.path.join(video_dir, video_key) for video_key in final_video_keys]
            self.videos += video_dirs
            self.labels += self._get_labels(video_dir, final_video_keys)
            self.mtype_index += [i for _ in range(len(final_video_keys))]

        if self.mode == 'train':
            self._oversample()
        else:
            self._get_clips()

    def _get_clips(self):
        for i, video_dir in enumerate(self.videos):
            frame_keys = sorted(os.listdir(video_dir))
            frame_count = len(frame_keys)
            num_samples = self.num_samples
            interval = self.interval # UNIFORM :1,2 / SPREAD: max(total_frames // num_samples, 1)
            max_length = (num_samples - 1) * self.interval + num_samples

            for starting_point in range(0, frame_count, (num_samples-1)*interval + num_samples):
                if (interval == 0) or (frame_count <= max_length):
                    sampled_keys = frame_keys[starting_point:starting_point+num_samples]
                else:
                    sampled_indices = np.arange(starting_point, frame_count, interval)[:num_samples]
                    sampled_keys = [frame_keys[idx] for idx in sampled_indices]

                if len(sampled_keys) < num_samples:
                    break

                self.clips += [sampled_keys]
                self.clip_src_idx.append(i)

    def __len__(self):
        if self.mode == 'train':
            return len(self.videos)
        else:
            return len(self.clips)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            video_dir = self.videos[index]
            frame_keys = sorted(os.listdir(video_dir))
            frame_count = len(frame_keys)
            clip_length = (self.num_samples - 1) * self.interval + self.num_samples

            if (self.interval == 0) or (frame_count <= clip_length):
                starting_point = random.randint(0, frame_count - self.num_samples)
                sampled_keys = frame_keys[starting_point:starting_point+self.num_samples]
            else:
                starting_point = random.randint(0, frame_count - clip_length)
                sampled_indices = np.arange(starting_point, frame_count, self.interval)[:self.num_samples]
                sampled_keys = [frame_keys[idx] for idx in sampled_indices]
        else:
            src_idx = self.clip_src_idx[index]
            video_dir = self.videos[src_idx]
            sampled_keys = self.clips[index]

        frames = []

        # Fix the randomness across the video for all frames
        state = torch.get_rng_state()   
        for frame_key in sampled_keys:
            frame = Image.open(os.path.join(video_dir, frame_key))
            torch.set_rng_state(state)
            frame = self.transforms(frame)
            frames.append(frame)

        frame_data = torch.stack(frames, dim=0).transpose(0,1)

        if self.mode == 'train':
            data = {'frame': frame_data, 'label': self.labels[index]}
        else:
            data = {'video': src_idx, 'frame': frame_data, 'label': self.labels[src_idx]}

        return data

class KeyPointDataset(Dataset):
    def __init__(self, path='../../kslr_metadata/train_keypoints', mode='train', transforms=None, **kwargs):
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
    ds = KeyPointDataset()

    for d in ds:
        print(d['keypoints'].shape)
        print("aa")