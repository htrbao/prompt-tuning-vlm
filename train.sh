CUDA_VISIBLE_DEVICES=3 python run_beit3_finetuning.py \
        --device cuda \
        --model beit3_large_patch16_480 \
        --input_size 480 \
        --task coco_captioning \
        --batch_size 95 \
        --eval_batch_size 32 \
        --layer_decay 1.0 \
        --lr 2e-4 \
        --randaug \
        --epochs 10 \
        --warmup_epochs 1 \
        --drop_path 0.1 \
        --sentencepiece_model ./model/beit3.spm \
        --finetune model/beit3_large_patch16_480_coco_captioning.pth \
        --data_path . \
        --output_dir output/large_notthing/ \
        --log_dir log/ \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 1 \
        --num_max_bpe_tokens 32 \
        --captioning_mask_prob 0.7 \
        --drop_worst_after 12000 \
        --dist_eval \
        --checkpoint_activations \
        --no_auto_resume

# CUDA_VISIBLE_DEVICES=3 python run_beit3_finetuning.py \
#         --device cuda \
#         --model beit3_large_patch16_480_with_gott \
#         --input_size 480 \
#         --task coco_captioning \
#         --batch_size 64 \
#         --eval_batch_size 32 \
#         --layer_decay 1.0 \
#         --lr 2e-3 \
#         --randaug \
#         --epochs 10 \
#         --warmup_epochs 1 \
#         --drop_path 0.1 \
#         --sentencepiece_model ./model/beit3.spm \
#         --finetune model/beit3_large_patch16_480_coco_captioning.pth \
#         --data_path . \
#         --output_dir output/large_soft/ \
#         --log_dir log/ \
#         --weight_decay 0.05 \
#         --seed 42 \
#         --save_ckpt_freq 5 \
#         --num_max_bpe_tokens 32 \
#         --captioning_mask_prob 0.7 \
#         --drop_worst_after 12000 \
#         --dist_eval \
#         --checkpoint_activations \
#         --no_auto_resume