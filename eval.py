import jieba
import os
import sys
import json
from collections import defaultdict
import statistics
import random
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from evaluate import load
from reformulation_experiment.utils import get_flickr8kcn_data

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
flickr8kcn_data = get_flickr8kcn_data()
for sample in flickr8kcn_data:
    non_tokenized_caption = ''.join(sample['caption'].split())
    tokenized_caption = ' '.join(list(jieba.cut(non_tokenized_caption, cut_all=False)))
    gt_data[sample['image_id']].append(tokenized_caption)

all_res = {}
candidate_image_ids = []
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
        if len(candidate_image_ids) < len(file_paths):
            candidate_image_ids.append(list(candidates.keys()))

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
    model_name = pattern.split('_infer_on_')[0].split('/')[-1]
    if len(all_res) == 0:
        all_res = {metric: {} for metric in res.keys()}
    for metric in res:
        if len(res[metric]) > 1:
            all_res[metric][model_name] = (statistics.mean(res[metric]), statistics.stdev(res[metric]))
        else:
            all_res[metric][model_name] = (res[metric][0], 0)

# Translated captions
translated_data_dir = 'reformulation_experiment/data/translated_data'
for file_name in os.listdir(translated_data_dir):
    translated_res = defaultdict(list)
    for cur_candidate_image_ids in candidate_image_ids:
        image_ids_dict = {x: True for x in cur_candidate_image_ids}
        with open(os.path.join(translated_data_dir, file_name)) as fp:
            cur_translated_data = json.load(fp)
        image_id_to_captions = defaultdict(list)
        for x in cur_translated_data:
            if x['image_id'] in image_ids_dict:
                image_id_to_captions[x['image_id']].append(x['caption'])
        image_id_to_caption = {x[0]: random.choice(x[1]) for x in image_id_to_captions.items()}
        candidates = {}
        for image_id, caption in image_id_to_caption.items():
            non_tokenized_caption = ''.join(caption.split())
            tokenized_caption = ' '.join(list(jieba.cut(non_tokenized_caption, cut_all=False)))
            candidates[image_id] = [tokenized_caption]
        cur_gt_data = {x[0]: x[1] for x in gt_data.items() if x[0] in candidates}
        metrics = compute_metrics(cur_gt_data, candidates)
        for metric_name, metric_res in metrics.items():
            translated_res[metric_name].append(metric_res)

    print('>>>>>>>>>>')
    print(file_name)
    for metric in translated_res:
        if len(translated_res[metric]) > 1:
            print(f'\t{metric}: {statistics.mean(translated_res[metric])} +- {statistics.stdev(translated_res[metric])}')
        else:
            print(f'\t{metric}: {translated_res[metric][0]}')
    print('<<<<<<<<<<')

    # Record for dumping
    for metric in translated_res:
        if len(translated_res[metric]) > 1:
            all_res[metric][file_name] = (statistics.mean(translated_res[metric]), statistics.stdev(translated_res[metric]))
        else:
            all_res[metric][file_name] = (translated_res[metric][0], 0)

with open('eval_res.json', 'w') as fp:
    fp.write(json.dumps(all_res))
