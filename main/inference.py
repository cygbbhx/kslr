import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import torchvision.transforms as T
from data_loader.data_loaders import evenly_sample_frames, uniform_sample_frames
import cv2
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import time
import os 

def vid2frames(path):
    cam = cv2.VideoCapture(path)
    vid_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    transforms = T.Compose([
                    T.CenterCrop(vid_height),  # Crop a square from the center
                    T.Resize((224, 224)),           # Resize to 224x224
                    T.ToTensor()
                ])

    clip = []
    frame_count = 0

    while True:
        ret, img = cam.read()
        if not ret: # no frame read, break
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        clip.append(transforms(Image.fromarray(img)))
        frame_count += 1

    
    sampled_indices = evenly_sample_frames(frame_count, 64, 0)
    sampled_frames = [clip[idx] for idx in sampled_indices]

    clip = torch.stack(sampled_frames, dim=0).transpose(0,1)
    clip = clip.unsqueeze(0)

    # print(clip.shape)
    return clip

def main(config, input_path):
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    # logger.info(model)

    logger.info('Loading checkpoint: {} ...\n'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # prepare lookup table
    lookup = pd.read_csv('demo/dictionary.csv')
    print("=> model prepared. start inference\n")

    correct = 0
    total = 0
    inf_times = []

    video_path = input_path

    with torch.no_grad():
        s = time.time()
        clip = vid2frames(video_path)
        outputs = model(clip.to(device))
        outputs = F.softmax(outputs, dim=-1)
        values, indices = torch.topk(outputs, k=5, dim=-1)

        # print(values, indices)

        pred_class = indices[0][0].cpu().item()
        pred_conf = values[0][0].cpu().item()

        # Comment this part if you do not have ground truth prepared
        gt_idx = int(video_path.split('/')[-1].split('_')[0])
        gt_lookup = lookup[lookup['word_idx']==gt_idx]
        gt_word = gt_lookup['word'].values[0]

        pred_idx = lookup.iloc[pred_class]['word_idx']
        pred_word = lookup.iloc[pred_class]['word']
        inf_time = (time.time()-s)

        print(f"predicted: {pred_idx:04}: {pred_word} ({pred_conf*100:.2f}%)")
        # Comment this part if you do not have ground truth prepared
        print(f"answer: {gt_idx:04}: {gt_word}")
        print(f"inference time: {inf_time:.4f}s")


        


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Inference')
    args.add_argument('-c', '--config', default='config/config_i3d.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='saved/models/Video_I3D_/1121_094552/checkpoint-epoch3.pth', type=str,
                      help='path to  weight to test (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--input', default='../../../inference_data/0001_1.mp4', type=str,
                      help='path to inference video')

    config = ConfigParser.from_args(args)

    args = args.parse_args()
    main(config, args.input)
