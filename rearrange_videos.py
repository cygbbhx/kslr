import os
import shutil
import zipfile
from tqdm import tqdm 

data_root = '2.Validation'
out_root = 'val_preprocessed'

zip_files = os.listdir(data_root)
real_word_zipfiles = [file for file in zip_files if 'real_word_video' in file and file.endswith('.zip')]

vocabs = []
with open('target_words.txt', 'r') as f:
    words = f.readlines()
    for word in words:
        vocabs.append(int(word))

os.makedirs(out_root, exist_ok=True)
for vocab in vocabs:
    os.makedirs(f'{out_root}/{vocab:04}', exist_ok=True)

video_vocabs = []
count = 0

for file in real_word_zipfiles:
    print(f'rearranging videos from {file}...')
    file_path = os.path.join(data_root, file)
    zf = zipfile.ZipFile(file_path)
    videos = [f for f in zf.namelist() if f.endswith('.mp4')]

    for video in tqdm(videos):
        video_dir = video[:video.rfind('/')]
        video_name = video.split('/')[-1]

        video_vocab = int(video_name.split('_')[2][4:])

        if video_vocab in vocabs:
            dst_path = f'{out_root}/{video_vocab:04}'
            zf.extract(video, dst_path)
            shutil.move(os.path.join(dst_path, video), os.path.join(dst_path, video_name))

            count += 1
            video_vocabs.append(video_vocab)

            os.rmdir(os.path.join(dst_path, video_dir))

            # remove all empty folders if there is any
            folders = [d for d in os.listdir(dst_path) if not d.endswith('.mp4')]
            for folder in folders:
                os.rmdir(os.path.join(dst_path, folder))

    print(f'{len(set(video_vocabs))} words found')
    print(f"done! total {count} videos filtered\n")
