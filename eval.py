import jieba
import sys
import json
import defaultdict
import statistics
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def compute_metrics(references, candidates):
    ###BLEU#####
    print("Compute BLEU ... ")
    pycoco_bleu = Bleu()
    bleu, _ = pycoco_bleu.compute_score(references, candidates)

    ####METEOR###
    print("Compute METEOR ... ")
    pycoco_meteor = Meteor()
    meteor, _ = pycoco_meteor.compute_score(references, candidates)
    del pycoco_meteor

    ####ROUGE###
    print("Compute ROUGE ... ")
    pycoco_rouge = Rouge()
    rouge, _ = pycoco_rouge.compute_score(references, candidates)

    ####CIDER###
    print("Compute CIDER ... ")
    pycoco_cider = Cider()
    cider, _ = pycoco_cider.compute_score(references, candidates)

    return {'bleu1': bleu[0], 'bleu2': bleu[1], 'bleu3': bleu[2], 'bleu4': bleu[3], 'meteor': meteor, 'rouge': rouge, 'cider': cider}

assert len(sys.argv) > 1, 'Please insert result files'
input_patterns = sys.argv[1:]
data = {}
for pattern in input_patterns:
    file_name = pattern.split('/')[-1]
    dir_path = '/'.join(pattern.split('/')[:-1])
    file_parts = file_name.split('@')
    assert len(file_parts) == 3
    options = file_parts[1].split(',')
    file_names = [file_parts[0] + x + file_parts[2] for x in options]
    file_paths = [os.path.join(dir_path, x) for x in file_names]
    
    data[pattern] = {}
    for file_path in file_paths:
        with open(file_path, 'r') as fp:
            data[pattern][file_path] = json.load(fp)

# Check that all results are on the same image ids
image_ids = None
for pattern, file_paths in data.items():
    for result in file_paths.values():
        cur_image_ids = [x['image_id'] for x in result]
        if image_ids is None:
            image_ids = cur_image_ids
        else:
            assert len(set(image_ids).intersection(cur_image_ids)) == len(image_ids), 'Not all results are on the same image ids'
image_ids_dict = {x: True for x in image_ids}

# Prepare gt captions
gt_data = defaultdict(list)
with open('datasets/flickr_caption.txt', 'r') as f:
    for line in fp:
        line = line.strip()
        image_id, caption = line.split('\t')
        if image_id in image_ids_dict:
            non_tokenized_caption = ''.join(caption.split())
            tokenized_caption = ' '.join(list(jieba.cut(non_tokenized_caption, cut_all=False)))
            gt_data[image_id].append(tokenized_caption)

for pattern, file_paths in data.items():
    res = defaultdict(list)
    for result in file_paths.values():
        candidates = {}
        for sample in result:
            image_id = sample['image_id']
            caption = sample['caption']
            non_tokenized_caption = ''.join(caption.split())
            tokenized_caption = ' '.join(list(jieba.cut(non_tokenized_caption, cut_all=False)))
            candidates[image_id] = [tokenized_caption]

        metrics = compute_metrics(gt_data, candidates)
        for metric_name, metric_res in metrics:
            res[metric_name].append(metric_res)
    print('>>>>>>>>>>')
    print(pattern)
    for metric in res:
        if len(res[metric]) > 1:
            print(f'\t{metric}: {statistics.mean(res[metric])} +- {statistics.stdev(res[metric])}')
        else:
            print(f'\t{metric}: {res[metric][0]}')
    print('<<<<<<<<<<')
