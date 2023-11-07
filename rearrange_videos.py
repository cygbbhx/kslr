import os
import shutil
from tqdm import tqdm 

data_root = 'dataset/01/'
videos = os.listdir(data_root)

vocabs = []
with open('target_words.txt', 'r') as f:
    words = f.readlines()
    for word in words:
        vocabs.append(int(word))

os.makedirs('preprocessed', exist_ok=True)

for vocab in vocabs:
    os.makedirs(f'preprocessed/{vocab}', exist_ok=True)

video_vocabs = set([int(v.split('_')[2][4:]) for v in videos])
print(f"rearranging videos... {len(video_vocabs)} words found")
count = 0

for video in tqdm(videos):
    vocab = int(video.split('_')[2][4:])
    if vocab in vocabs:
        source_video = os.path.join(data_root, video)
        shutil.copy(source_video, f'preprocessed/{vocab:04}/{video}')
        count += 1

print(f"done! total {count} videos filtered")
