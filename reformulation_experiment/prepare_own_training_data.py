import json
import sys

assert len(sys.argv) == 2
exp_ind = int(sys.argv[1])

with open(f'reformulation_experiment/data/infer/base_infer_on_additional_train_{exp_ind}.json', 'r') as fp:
    data = json.load(fp)

with open(f'reformulation_experiment/data/own_train_data/own_train_data_{exp_ind}.txt', 'w') as fp:
    for sample in data:
        fp.write(f'{sample["image_id"]}\t{sample["caption"]}\n')
