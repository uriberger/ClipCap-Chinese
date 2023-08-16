import json
import sys

assert len(sys.argv) == 4
input_file = sys.argv[1]
output_file = sys.argv[2]
exp_ind = int(sys.argv[3])

with open(f'reformulation_experiment/data/image_ids/additional_train_image_ids_{exp_ind}.json', 'r') as fp:
    image_ids = json.load(fp)
image_ids_dict = {x: True for x in image_ids}

with open(input_file, 'r') as fp:
    data = json.load(fp)

with open(output_file, 'w') as fp:
    for sample in data:
        if sample['image_id'] in image_ids_dict:
            fp.write(f'{sample["image_id"]}\t{sample["caption"]}\n')
