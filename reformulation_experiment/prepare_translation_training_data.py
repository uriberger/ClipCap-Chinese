from collections import defaultdict
import json
import sys
import random

assert len(sys.argv) == 2
exp_ind = int(sys.argv[1])

with open(f'reformulation_experiment/data/image_ids/additional_train_image_ids_{exp_ind}.json', 'r') as fp:
    image_ids = json.load(fp)
image_ids_dict = {x: True for x in image_ids}

with open('reformulation_experiment/data/translated_data/flickr30k_zh_translated_helsinki.json', 'r') as fp:
    translated_data = json.load(fp)
image_id_to_captions = defaultdict(list)
for x in translated_data:
    image_id_to_captions[x['image_id']].append(x['caption'])


with open(f'reformulation_experiment/data/translated_train_data/translated_train_data_{exp_ind}.txt', 'w') as fp:
    for image_id, captions in image_id_to_captions.items():
        if image_id in image_ids_dict:
            fp.write(f'{image_id}\t{random.choice(captions)}\n')
