from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
import math
import json
import os
import argparse

def translate(sentences, source_language, target_language, output_file, batch_size=64):
    model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language}'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)

    output_file_name = output_file + '.json'
    '''if os.path.isfile(output_file_name):
        with open(output_file_name, 'r') as fp:
            res = json.load(fp)
        sentences = sentences[len(res):]
    else:
        res = []'''
    res = []
    batch_start = 0
    batch_ind = 0
    batch_num = math.ceil(len(sentences)/batch_size)
    t = time.time()
    while batch_start < len(sentences):
        if batch_ind % 100 == 0:
            print('Starting batch ' + str(batch_ind) + ' out of ' + str(batch_num) + ', time from prev ' + str(time.time() - t), flush=True)
            with open(output_file_name, 'w') as fp:
                fp.write(json.dumps(res))
            t = time.time()
        batch_end = min(batch_start + batch_size, len(sentences))
        batch = sentences[batch_start:batch_end]
        inputs = tokenizer(batch, return_tensors='pt', padding=True).to(device)
        outputs = model.generate(**inputs, num_beams=5, num_return_sequences=1)
        res += [tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(len(batch))]
        batch_start = batch_end
        batch_ind += 1

    print('Writing result to ' + output_file_name, flush=True)
    with open(output_file_name, 'w') as fp:
        fp.write(json.dumps(res))
    print('Finished!')

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_language', required=True)
    parser.add_argument('--target_language', required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--output_format', required=True)
    args = parser.parse_args()

    with open(args.input_file, 'r') as fp:
        data = json.load(fp)
    sentences = [x['caption'] for x in data]    
    translated = translate(sentences, args.source_language, args.target_language, args.output_file)
    if args.output_format == 'caption':
        res = [{'image_id': data[i]['image_id'], 'caption': translated[i]} for i in range(len(data))]
    elif args.output_format == 'image':
        res = [{'image_id': data[i]['image_id'], 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/train2014/COCO_train2014_{str(data[i]["image_id"]).zfill(12)}.jpg', 'sentences': [{'raw': translated[i]}]} for i in range(len(data))]
    else:
        assert False
    with open(args.output_file + '.json', 'w') as fp:
        fp.write(json.dumps(res))
