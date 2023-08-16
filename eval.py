import jieba
import os
import sys
import json
from collections import defaultdict
import statistics
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from evaluate import load

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

    ####BERTScore###
    bertscore = load("bertscore")
    reference_list = list(references.items())
    reference_list.sort(key=lambda x:x[0])
    references = [x[1] for x in reference_list]
    prediction_list = list(candidates.items())
    prediction_list.sort(key=lambda x:x[0])
    predictions = [x[1][0] for x in prediction_list]
    results = bertscore.compute(predictions=predictions, references=references, lang="zh")
    bertscore = statistics.mean(results['f1'])

    return {'bleu1': bleu[0], 'bleu2': bleu[1], 'bleu3': bleu[2], 'bleu4': bleu[3], 'meteor': meteor, 'rouge': rouge, 'cider': cider, 'bertscore': bertscore}

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

# Prepare gt captions
gt_data = defaultdict(list)
with open('datasets/flickr_caption.txt', 'r') as fp:
    for line in fp:
        line = line.strip()
        image_id, caption = line.split('\t')
        image_id = int(image_id)
        non_tokenized_caption = ''.join(caption.split())
        tokenized_caption = ' '.join(list(jieba.cut(non_tokenized_caption, cut_all=False)))
        gt_data[image_id].append(tokenized_caption)

all_res = {}
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

        cur_gt_data = {x[0]: x[1] for x in gt_data.items() if x[0] in candidates}

        metrics = compute_metrics(cur_gt_data, candidates)

        for metric_name, metric_res in metrics.items():
            res[metric_name].append(metric_res)
    print('>>>>>>>>>>')
    print(pattern)
    for metric in res:
        if len(res[metric]) > 1:
            print(f'\t{metric}: {statistics.mean(res[metric])} +- {statistics.stdev(res[metric])}')
        else:
            print(f'\t{metric}: {res[metric][0]}')
    print('<<<<<<<<<<')

    # Record for dumping
    model_name = pattern.split('_infer_on_')[0]
    if len(all_res) == 0:
        all_res = {metric: {} for metric in res.keys()}
    for metric in res:
        if len(res[metric]) > 1:
            all_res[metric][model_name] = (statistics.mean(res[metric]), statistics.stdev(res[metric]))
        else:
            all_res[metric][model_name] = (res[metric][0], 0)
with open('eval_res.json', 'w') as fp:
    fp.write(json.dumps(all_res))
