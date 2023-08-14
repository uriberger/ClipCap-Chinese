import json
import sys
import random
from utils import get_flickr_captions

assert len(sys.argv) == 3
exp_ind = int(sys.argv[1])
base_train_sample_num = int(sys.argv[2])

image_id_to_captions = get_flickr_captions()

print(f'Collected {len(image_id_to_captions)} images and {sum([len(x) for x in image_id_to_captions.values()])} captions')

image_ids = list(image_id_to_captions.keys())

test_image_num = 1000
test_image_ids = random.sample(image_ids, test_image_num)
test_image_dict = {x: True for x in test_image_ids}

train_image_ids = [x for x in image_ids if x not in test_image_dict]
base_image_ids = random.sample(train_image_ids, base_train_sample_num)

base_image_ids_dict = {x: True for x in base_image_ids}
additional_image_ids = [x for x in train_image_ids if x not in base_image_ids_dict]
additional_image_ids_dict = {x: True for x in additional_image_ids}

with open(f'reformulation_experiment/data/image_ids/base_train_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(base_image_ids))
with open(f'reformulation_experiment/data/image_ids/additional_train_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(additional_image_ids))
with open(f'reformulation_experiment/data/image_ids/test_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(test_image_ids))
print(f'Selected {len(base_image_ids)} base train image ids, {len(additional_image_ids)} additional train image ids and {len(test_image_ids)} test image ids')

base_train_fp = open(f'reformulation_experiment/data/base_train_data/base_train_data_{exp_ind}.txt', 'w')
additional_train_fp = open(f'reformulation_experiment/data/gt_train_data/gt_train_data_{exp_ind}.txt', 'w')
for image_id, captions in image_id_to_captions.items():
    cur_fp = None
    if image_id in base_image_ids_dict:
        cur_fp = base_train_fp
    elif image_id in additional_image_ids_dict:
        cur_fp = additional_train_fp
        captions = random.sample(captions, 1)
    if cur_fp is not None:
        for caption in captions:
            cur_fp.write(f'{image_id}\t{caption}\n')
base_train_fp.close()
additional_train_fp.close()
