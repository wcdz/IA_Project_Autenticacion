# libraries
import os
import random
import shutil
from itertools import islice

# folders paths
OUT_PUT_FOLDER_PATH = 'objectDetection/SplitData'
INPUT_FOLDER_PATH = 'objectDetection/All'
split_ratio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ['Gafas',
           'Gorras',
           'Casaca',
           'Polo',
           'Pantalones',
           'Shorts',
           'Falda',
           'Vestido',
           'Mochila',
           'Zapatos']

try:
    shutil.rmtree(OUT_PUT_FOLDER_PATH)
    print('Clean SplitData')
except:
    os.mkdir(OUT_PUT_FOLDER_PATH)

# Create Folder
os.makedirs(f'{OUT_PUT_FOLDER_PATH}/train/images', exist_ok=True)
os.makedirs(f'{OUT_PUT_FOLDER_PATH}/train/labels', exist_ok=True)
os.makedirs(f'{OUT_PUT_FOLDER_PATH}/val/images', exist_ok=True)
os.makedirs(f'{OUT_PUT_FOLDER_PATH}/val/labels', exist_ok=True)
os.makedirs(f'{OUT_PUT_FOLDER_PATH}/test/images', exist_ok=True)
os.makedirs(f'{OUT_PUT_FOLDER_PATH}/test/labels', exist_ok=True)

# Img Names
list_names = os.listdir(INPUT_FOLDER_PATH)
unique_names = []
for name in list_names:
    unique_names.append(name.split('.')[0])

unique_names = list(set(unique_names))
print(unique_names)

# Shuffle
random.shuffle(unique_names)

# Img number folders
len_data = len(unique_names)
len_train = int(len_data * split_ratio['train'])
len_val = int(len_data * split_ratio['val'])
len_test = int(len_data * split_ratio['test'])

print(f'Total images: {len_data}\nSplit Train: {len_train}\nSplit Val: {len_val}\nSplit Test: {len_test}')

# Img train
if len_data != (len_train + len_val + len_test):
    remaining = len_data - (len_train + len_val + len_test)
    len_train += remaining

# Split
len_split = [len_train, len_val, len_test]
input = iter(list_names)
output = [list(islice(input, elem)) for elem in len_split]
print(output)

# Copy
sequence = ['train', 'val', 'test']
for i, out in enumerate(output):
    for fileName in out:
        split_filename = fileName.rsplit('.', 1)[0]
        shutil.copy(f'{INPUT_FOLDER_PATH}/{split_filename}.jpg',
                    f'{OUT_PUT_FOLDER_PATH}/{sequence[i]}/images/{split_filename}.jpg')
        shutil.copy(f'{INPUT_FOLDER_PATH}/{split_filename}.txt',
                    f'{OUT_PUT_FOLDER_PATH}/{sequence[i]}/labels/{split_filename}.txt')

print('Split complete')

# Data.yaml
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'

f = open(f"{OUT_PUT_FOLDER_PATH}/Dataset.yaml", 'a')
f.write(dataYaml)
f.close()

print("DATASET.YAML -> complete")
