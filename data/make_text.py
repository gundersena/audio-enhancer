import os
import random
import librosa

file_path = 'VCTK-Corpus/wav48'
dataset_path = './multispeaker'

folders = os.listdir(file_path)

for folder in folders:
    if folder == '.DS_Store':
        folders.remove(folder)
print (folders)

for folder in folders:
    files = os.listdir(os.path.join(file_path, folder))
    random.shuffle(files)
    for file in files:
        if len(x) < 10:
            input('This file is corrupted')
        u = random.uniform(0,1) # a single value is returned between 0 and 1
        if u > 0.1:
            with open(f'{dataset_path}/train-files1.txt', 'a+') as text_file:
                text_file.write(f'{file_path}/{folder}/{file}')
                text_file.write('\n')
        if u < 0.1:
            with open(f'{dataset_path}/val-files1.txt', 'a+') as text_file:
                text_file.write(f'{file_path}/{folder}/{file}')
                text_file.write('\n')
