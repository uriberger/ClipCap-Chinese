#!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=reformulation_experiment

if [[ -z "${EXP_IND}" ]]; then
    EXP_IND=0
fi

echo "Experiment ${EXP_IND}"

# Base training
echo "$MSG_PREFIX Prepare base training data"
venv/bin/python ${BASE_DIR}/prepare_base_training_data_coco.py ${EXP_IND}
echo "$MSG_PREFIX Base preprocess"
venv/bin/python process_flickr.py --clip_model_path ViT-B/32 --image_path /cs/labs/oabend/uriber/datasets/flickr30/images --caption_path ${BASE_DIR}/data/base_train_data/base_train_data_${EXP_IND}.txt --output_path ${BASE_DIR}/data/base_train_data/preprocessed_data_${EXP_IND}.pkl
echo "$MSG_PREFIX Base training"
venv/bin/python train.py --data_path ${BASE_DIR}/data/base_train_data/preprocessed_data_${EXP_IND}.pkl --gpt2_path pretrain_models/gpt2 --bert_path pretrain_models/bert --output_path ${BASE_DIR}/output/exp_${EXP_IND}_base --lr 2e-5 --epochs 10 --prefix_len 10 --constant_len 10 --clip_size 512 --bs_train 40 --dev_size 100 --bs_eval 10 --max_len 100 --warmup_steps 5000 --eval_step 500 --finetune_gpt2 --mapping_type mlp --do_train
echo "$MSG_PREFIX Base inference"
venv/bin/python predict.py --image_ids_path ${BASE_DIR}/data/image_ids/test_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path ${BASE_DIR}/data/infer/base_infer_on_test_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp

# Translation based training
echo "$MSG_PREFIX Prepare translated training data"
venv/bin/python ${BASE_DIR}/prepare_translation_training_data_coco.py ${EXP_IND}
echo "$MSG_PREFIX Translation preprocess"
venv/bin/python process_flickr.py --clip_model_path ViT-B/32 --train_set_file ${BASE_DIR}/data/translated_train_data/translated_train_data_${EXP_IND}.json --output_path ${BASE_DIR}/data/translated_train_data/preprocessed_data_${EXP_IND}.pkl
echo "$MSG_PREFIX Translation training"
venv/bin/python train.py --data_path ${BASE_DIR}/data/translated_train_data/preprocessed_data_${EXP_IND}.pkl --gpt2_path pretrain_models/gpt2 --bert_path pretrain_models/bert --output_path ${BASE_DIR}/output/exp_${EXP_IND}_translated --lr 2e-5 --epochs 10 --prefix_len 10 --constant_len 10 --clip_size 512 --bs_train 40 --dev_size 100 --bs_eval 10 --max_len 100 --warmup_steps 5000 --eval_step 500 --finetune_gpt2 --mapping_type mlp --do_train --load_path ${BASE_DIR}/output/exp_${EXP_IND}_base/checkpoint-last.pt
echo "$MSG_PREFIX Translation inference"
venv/bin/python predict.py --image_ids_path ${BASE_DIR}/data/image_ids/test_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path ${BASE_DIR}/data/infer/translated_infer_on_test_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp

# Own captions based training
echo "$MSG_PREFIX Base inference on additional train"
venv/bin/python predict.py --image_ids_path ${BASE_DIR}/data/base_train_data/additional_train_data_new_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp
echo "$MSG_PREFIX Prepare own training data"
venv/bin/python ${BASE_DIR}/convert_to_training_data_coco.py ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}.json ${BASE_DIR}/data/own_train_data/own_train_data_${EXP_IND}.json ${EXP_IND}
echo "$MSG_PREFIX Own preprocess"
venv/bin/python process_flickr.py --clip_model_path ViT-B/32 --image_path /cs/labs/oabend/uriber/datasets/flickr30/images --caption_path ${BASE_DIR}/data/own_train_data/own_train_data_${EXP_IND}.json --output_path ${BASE_DIR}/data/own_train_data/preprocessed_data_${EXP_IND}.pkl
echo "$MSG_PREFIX Own training"
venv/bin/python train.py --data_path ${BASE_DIR}/data/own_train_data/preprocessed_data_${EXP_IND}.pkl --gpt2_path pretrain_models/gpt2 --bert_path pretrain_models/bert --output_path ${BASE_DIR}/output/exp_${EXP_IND}_own --lr 2e-5 --epochs 10 --prefix_len 10 --constant_len 10 --clip_size 512 --bs_train 40 --dev_size 100 --bs_eval 10 --max_len 100 --warmup_steps 5000 --eval_step 500 --finetune_gpt2 --mapping_type mlp --do_train --load_path ${BASE_DIR}/output/exp_${EXP_IND}_base/checkpoint-last.pt
echo "$MSG_PREFIX Own inference"
venv/bin/python predict.py --image_ids_path ${BASE_DIR}/data/image_ids/test_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path ${BASE_DIR}/data/infer/own_infer_on_test_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp

# Reformulations based training
echo "$MSG_PREFIX zh->en"
venv/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}.json --output_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_translated --source_language zh --target_language en --output_format caption
venv/bin/python prepare_data_before_reformulation.py ${EXP_IND}
echo "$MSG_PREFIX Reformulation"
rm -f ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated_flickr.json
rm -f ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated_coco.json
cd ../AliceMind/mPLUG
venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file ../../ClipCap-Chinese/${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_translated_flickr.json --output_format caption --output_file ../../ClipCap-Chinese/${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated_flickr --dataset flickr30k
venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file ../../ClipCap-Chinese/${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_translated_coco.json --output_format caption --output_file ../../ClipCap-Chinese/${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated_coco --dataset COCO
cd ../../ClipCap-Chinese
echo "$MSG_PREFIX en->zh"
venv/bin/python prepare_data_after_reformulation.py ${EXP_IND}
venv/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated.json --output_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_reformulated --source_language en --target_language zh --output_format caption
echo "$MSG_PREFIX Prepare reformulations training data"
venv/bin/python ${BASE_DIR}/convert_to_training_data_coco.py ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_reformulated.json ${BASE_DIR}/data/reformulations_train_data/reformulations_train_data_${EXP_IND}.json ${EXP_IND}
echo "$MSG_PREFIX Reformulations preprocess"
venv/bin/python process_flickr.py --clip_model_path ViT-B/32 --image_path /cs/labs/oabend/uriber/datasets/flickr30/images --caption_path ${BASE_DIR}/data/reformulations_train_data/reformulations_train_data_${EXP_IND}.json --output_path ${BASE_DIR}/data/reformulations_train_data/preprocessed_data_${EXP_IND}.pkl
echo "$MSG_PREFIX Reformulations training"
venv/bin/python train.py --data_path ${BASE_DIR}/data/reformulations_train_data/preprocessed_data_${EXP_IND}.pkl --gpt2_path pretrain_models/gpt2 --bert_path pretrain_models/bert --output_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulations --lr 2e-5 --epochs 10 --prefix_len 10 --constant_len 10 --clip_size 512 --bs_train 40 --dev_size 100 --bs_eval 10 --max_len 100 --warmup_steps 5000 --eval_step 500 --finetune_gpt2 --mapping_type mlp --do_train --load_path ${BASE_DIR}/output/exp_${EXP_IND}_base/checkpoint-last.pt
echo "$MSG_PREFIX Reformulations inference"
venv/bin/python predict.py --image_ids_path ${BASE_DIR}/data/image_ids/test_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulations/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path ${BASE_DIR}/data/infer/reformulations_infer_on_test_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp

# STILL NOT IMPLEMENTED
# MPLUG RE Based training
#echo "$MSG_PREFIX Prepare mplug re training data"
#venv/bin/python ${BASE_DIR}/convert_to_training_data_coco.py ${BASE_DIR}/data/translated_data/flickr30k_mplug_re_zh_translated_helsinki.json ${BASE_DIR}/data/translated_train_data/mplug_re_train_data_${EXP_IND}.txt ${EXP_IND}
#echo "$MSG_PREFIX mPLUG re preprocess"
#venv/bin/python process_flickr.py --clip_model_path ViT-B/32 --image_path /cs/labs/oabend/uriber/datasets/flickr30/images --caption_path ${BASE_DIR}/data/translated_train_data/mplug_re_train_data_${EXP_IND}.txt --output_path ${BASE_DIR}/data/translated_train_data/mplug_re_preprocessed_data_${EXP_IND}.pkl
#echo "$MSG_PREFIX mPLUG re training"
#venv/bin/python train.py --data_path ${BASE_DIR}/data/translated_train_data/mplug_re_preprocessed_data_${EXP_IND}.pkl --gpt2_path pretrain_models/gpt2 --bert_path pretrain_models/bert --output_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re --lr 2e-5 --epochs 10 --prefix_len 10 --constant_len 10 --clip_size 512 --bs_train 40 --dev_size 100 --bs_eval 10 --max_len 100 --warmup_steps 5000 --eval_step 500 --finetune_gpt2 --mapping_type mlp --do_train --load_path ${BASE_DIR}/output/exp_${EXP_IND}_base/checkpoint-last.pt
#echo "$MSG_PREFIX mPLUG re inference"
#venv/bin/python predict.py --image_ids_path ${BASE_DIR}/data/image_ids/test_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path ${BASE_DIR}/data/infer/mplug_re_infer_on_test_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp

# MPLUG Based training
#echo "$MSG_PREFIX Prepare mplug training data"
#venv/bin/python ${BASE_DIR}/convert_to_training_data.py ${BASE_DIR}/data/translated_data/flickr30k_mplug_zh_translated_helsinki.json ${BASE_DIR}/data/translated_train_data/mplug_train_data_${EXP_IND}.txt ${EXP_IND}
#echo "$MSG_PREFIX mPLUG preprocess"
#venv/bin/python process_flickr.py --clip_model_path ViT-B/32 --image_path /cs/labs/oabend/uriber/datasets/flickr30/images --caption_path ${BASE_DIR}/data/translated_train_data/mplug_train_data_${EXP_IND}.txt --output_path ${BASE_DIR}/data/translated_train_data/mplug_preprocessed_data_${EXP_IND}.pkl
#echo "$MSG_PREFIX mPLUG training"
#venv/bin/python train.py --data_path ${BASE_DIR}/data/translated_train_data/mplug_preprocessed_data_${EXP_IND}.pkl --gpt2_path pretrain_models/gpt2 --bert_path pretrain_models/bert --output_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug --lr 2e-5 --epochs 10 --prefix_len 10 --constant_len 10 --clip_size 512 --bs_train 40 --dev_size 100 --bs_eval 10 --max_len 100 --warmup_steps 5000 --eval_step 500 --finetune_gpt2 --mapping_type mlp --do_train --load_path ${BASE_DIR}/output/exp_${EXP_IND}_base/checkpoint-last.pt
#echo "$MSG_PREFIX mPLUG inference"
#venv/bin/python predict.py --image_ids_path ${BASE_DIR}/data/image_ids/test_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path ${BASE_DIR}/data/infer/mplug_infer_on_test_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp

# BLIP Based training
#echo "$MSG_PREFIX Prepare BLIP training data"
#venv/bin/python ${BASE_DIR}/convert_to_training_data.py ${BASE_DIR}/data/translated_data/flickr30k_blip_zh_translated_helsinki.json ${BASE_DIR}/data/translated_train_data/blip_train_data_${EXP_IND}.txt ${EXP_IND}
#echo "$MSG_PREFIX BLIP preprocess"
#venv/bin/python process_flickr.py --clip_model_path ViT-B/32 --image_path /cs/labs/oabend/uriber/datasets/flickr30/images --caption_path ${BASE_DIR}/data/translated_train_data/blip_train_data_${EXP_IND}.txt --output_path ${BASE_DIR}/data/translated_train_data/blip_preprocessed_data_${EXP_IND}.pkl
#echo "$MSG_PREFIX BLIP training"
#venv/bin/python train.py --data_path ${BASE_DIR}/data/translated_train_data/blip_preprocessed_data_${EXP_IND}.pkl --gpt2_path pretrain_models/gpt2 --bert_path pretrain_models/bert --output_path ${BASE_DIR}/output/exp_${EXP_IND}_blip --lr 2e-5 --epochs 10 --prefix_len 10 --constant_len 10 --clip_size 512 --bs_train 40 --dev_size 100 --bs_eval 10 --max_len 100 --warmup_steps 5000 --eval_step 500 --finetune_gpt2 --mapping_type mlp --do_train --load_path ${BASE_DIR}/output/exp_${EXP_IND}_base/checkpoint-last.pt
#echo "$MSG_PREFIX BLIP inference"
#venv/bin/python predict.py --image_ids_path ${BASE_DIR}/data/image_ids/test_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_blip/checkpoint-last.pt --gpt2_model_path pretrain_models/gpt2 --bert_model_path pretrain_models/bert --clip_model_path ViT-B/32 --output_path ${BASE_DIR}/data/infer/blip_infer_on_test_${EXP_IND}.json --prefix_len 10 --constant_len 10 --clip_size 512 --max_len 100 --batch_size 4 --temperature 1 --topk 0 --topp 0.8 --num_generate 1 --finetune_gpt2 --mapping_type mlp

echo "$MSG_PREFIX Finished"
