import json
import sys

assert len(sys.argv) == 4
input_file = sys.argv[1]
output_file = sys.argv[2]
exp_ind = int(sys.argv[3])

with open(f'reformulation_experiment/data/base_train_data/additional_train_data_{exp_ind}.json', 'r') as fp:
    additional_train_data = json.load(fp)

with open(input_file, 'r') as fp:
    data = json.load(fp)

train_data = []
for sample in data:
    train_data.append({'image_id': sample['image_id'], 'image_path': additional_train_data[sample['image_id']]['file_path'], 'caption': sample['caption']})
with open(output_file, 'w') as fp:
    output_file.write(json.dumps(train_data))
