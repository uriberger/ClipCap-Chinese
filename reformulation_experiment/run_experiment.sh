#!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=reformulation_experiment
EXP_IND=0
BASE_SAMPLE_NUM=10000

echo "Experiment ${EXP_IND}, base training sample num ${BASE_SAMPLE_NUM}"

# Base training
echo "$MSG_PREFIX Prepare base training data"
venv/bin/python ${BASE_DIR}/prepare_base_training_data.py ${EXP_IND} ${BASE_SAMPLE_NUM}
echo "$MSG_PREFIX Base preprocess"
venv/bin/python process_flickr.py --clip_model_path ViT-B/32 --image_path /cs/labs/oabend/uriber/datasets/flickr30/images --caption_path reformulation_experiment/data/base_train_data/base_train_data_${EXP_IND}.txt --output_path reformulation_experiment/data/base_train_data/preprocessed_data_${EXP_IND}.pkl
echo "$MSG_PREFIX Base training"
venv/bin/python train.py --data_path reformulation_experiment/data/base_train_data/preprocessed_data_${EXP_IND}.pkl --gpt2_path pretrain_models/gpt2 --bert_path pretrain_models/bert --output_path reformulation_experiment/output/exp_${EXP_IND}_base --lr 2e-5 --epochs 10 --prefix_len 10 --constant_len 10 --clip_size 512 --bs_train 40 --dev_size 100 --bs_eval 10 --max_len 100 --warmup_steps 5000 --eval_step 500 --finetune_gpt2 --mapping_type mlp --do_train
echo "$MSG_PREFIX Base inference"
venv/bin/python predict.py --image_ids_path reformulation_experiment/data/image_ids/test_image_ids_${EXP_IND}.json --model_path reformulation_experiment/output/exp_${EXP_IND}_base/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path reformulation_experiment/data/infer/base_infer_on_test_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp

# GT based training
echo "$MSG_PREFIX GT preprocess"
venv/bin/python process_flickr.py --clip_model_path ViT-B/32 --image_path /cs/labs/oabend/uriber/datasets/flickr30/images --caption_path reformulation_experiment/data/gt_train_data/gt_train_data_${EXP_IND}.txt --output_path reformulation_experiment/data/gt_train_data/preprocessed_data_${EXP_IND}.pkl
echo "$MSG_PREFIX GT training"
venv/bin/python train.py --data_path reformulation_experiment/data/gt_train_data/preprocessed_data_${EXP_IND}.pkl --gpt2_path pretrain_models/gpt2 --bert_path pretrain_models/bert --output_path reformulation_experiment/output/exp_${EXP_IND}_gt --lr 2e-5 --epochs 10 --prefix_len 10 --constant_len 10 --clip_size 512 --bs_train 40 --dev_size 100 --bs_eval 10 --max_len 100 --warmup_steps 5000 --eval_step 500 --finetune_gpt2 --mapping_type mlp --do_train --load_path reformulation_experiment/output/exp_${EXP_IND}_base/checkpoint-last.pt
echo "$MSG_PREFIX GT inference"
venv/bin/python predict.py --image_ids_path reformulation_experiment/data/image_ids/test_image_ids_${EXP_IND}.json --model_path reformulation_experiment/output/exp_${EXP_IND}_gt/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path reformulation_experiment/data/infer/gt_infer_on_test_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp

# Translation based training
echo "$MSG_PREFIX Prepare translated training data"
venv/bin/python reformulation_experiment/prepare_translation_train_data.py ${EXP_IND}
echo "$MSG_PREFIX Translation preprocess"
venv/bin/python process_flickr.py --clip_model_path ViT-B/32 --image_path /cs/labs/oabend/uriber/datasets/flickr30/images --caption_path reformulation_experiment/data/translated_train_data/translated_train_data_${EXP_IND}.txt --output_path reformulation_experiment/data/translated_train_data/preprocessed_data_${EXP_IND}.pkl
echo "$MSG_PREFIX Translation training"
venv/bin/python train.py --data_path reformulation_experiment/data/translated_train_data/preprocessed_data_${EXP_IND}.pkl --gpt2_path pretrain_models/gpt2 --bert_path pretrain_models/bert --output_path reformulation_experiment/output/exp_${EXP_IND}_translated --lr 2e-5 --epochs 10 --prefix_len 10 --constant_len 10 --clip_size 512 --bs_train 40 --dev_size 100 --bs_eval 10 --max_len 100 --warmup_steps 5000 --eval_step 500 --finetune_gpt2 --mapping_type mlp --do_train --load_path reformulation_experiment/output/exp_${EXP_IND}_base/checkpoint-last.pt
echo "$MSG_PREFIX Translation inference"
venv/bin/python predict.py --image_ids_path reformulation_experiment/data/image_ids/test_image_ids_${EXP_IND}.json --model_path reformulation_experiment/output/exp_${EXP_IND}_translated/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path reformulation_experiment/data/infer/translated_infer_on_test_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp

# Own captions based training
echo "$MSG_PREFIX Base inference on val"
venv/bin/python predict.py --image_ids_path reformulation_experiment/data/image_ids/additional_train_image_ids_${EXP_IND}.json --model_path reformulation_experiment/output/exp_${EXP_IND}_base/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path reformulation_experiment/data/infer/base_infer_on_additional_train_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp
echo "$MSG_PREFIX Prepare own training data"
venv/bin/python reformulation_experiment/prepare_own_train_data.py ${EXP_IND}
echo "$MSG_PREFIX Own preprocess"
venv/bin/python process_flickr.py --clip_model_path ViT-B/32 --image_path /cs/labs/oabend/uriber/datasets/flickr30/images --caption_path reformulation_experiment/data/own_train_data/own_train_data_${EXP_IND}.txt --output_path reformulation_experiment/data/own_train_data/preprocessed_data_${EXP_IND}.pkl
echo "$MSG_PREFIX Own training"
venv/bin/python train.py --data_path reformulation_experiment/data/own_train_data/preprocessed_data_${EXP_IND}.pkl --gpt2_path pretrain_models/gpt2 --bert_path pretrain_models/bert --output_path reformulation_experiment/output/exp_${EXP_IND}_own --lr 2e-5 --epochs 10 --prefix_len 10 --constant_len 10 --clip_size 512 --bs_train 40 --dev_size 100 --bs_eval 10 --max_len 100 --warmup_steps 5000 --eval_step 500 --finetune_gpt2 --mapping_type mlp --do_train --load_path reformulation_experiment/output/exp_${EXP_IND}_base/checkpoint-last.pt
echo "$MSG_PREFIX Own inference"
venv/bin/python predict.py --image_ids_path reformulation_experiment/data/image_ids/test_image_ids_${EXP_IND}.json --model_path reformulation_experiment/output/exp_${EXP_IND}_own/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path reformulation_experiment/data/infer/own_infer_on_test_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp

# Reformulations based training
echo "$MSG_PREFIX zh->en"
venv/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}.json --output_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_translated --source_language zh --target_language en --output_format caption
echo "$MSG_PREFIX Reformulation"
rm -f ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated.json
cd ../AliceMind/mPLUG
venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file ../../ClipCap-Chinese/${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND}_en_translated.json --output_format caption --output_file ../../ClipCap-Chinese/${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated --dataset flickr30k
cd ../../ClipCap-Chinese
echo "$MSG_PREFIX en->zh"
venv/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated.json --output_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_reformulated --source_language en --target_language zh --output_format caption
echo "$MSG_PREFIX Prepare reformulations training data"
venv/bin/python reformulation_experiment/prepare_reformulation_train_data.py ${EXP_IND}
echo "$MSG_PREFIX Reformulations preprocess"
venv/bin/python process_flickr.py --clip_model_path ViT-B/32 --image_path /cs/labs/oabend/uriber/datasets/flickr30/images --caption_path reformulation_experiment/data/reformulations_train_data/reformulations_train_data_${EXP_IND}.txt --output_path reformulation_experiment/data/reformulations_train_data/preprocessed_data_${EXP_IND}.pkl
echo "$MSG_PREFIX Reformulations training"
venv/bin/python train.py --data_path reformulation_experiment/data/reformulations_train_data/preprocessed_data_${EXP_IND}.pkl --gpt2_path pretrain_models/gpt2 --bert_path pretrain_models/bert --output_path reformulation_experiment/output/exp_${EXP_IND}_reformulations --lr 2e-5 --epochs 10 --prefix_len 10 --constant_len 10 --clip_size 512 --bs_train 40 --dev_size 100 --bs_eval 10 --max_len 100 --warmup_steps 5000 --eval_step 500 --finetune_gpt2 --mapping_type mlp --do_train --load_path reformulation_experiment/output/exp_${EXP_IND}_base/checkpoint-last.pt
echo "$MSG_PREFIX Reformulations inference"
venv/bin/python predict.py --image_ids_path reformulation_experiment/data/image_ids/test_image_ids_${EXP_IND}.json --model_path reformulation_experiment/output/exp_${EXP_IND}_reformulations/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path reformulation_experiment/data/infer/reformulations_infer_on_test_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp

echo "$MSG_PREFIX Finished"
