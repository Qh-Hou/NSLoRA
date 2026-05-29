DATA_ROOT="${CIFAR100_ROOT:-${DATA_ROOT:-DATA_ROOT}}"

python train_eval.py -d cifar100 -t 10 \
    -m vit_base_patch16_clip_quickgelu_224.openai --head_dim_type text_dim --logit_type sim_imgtext --transform_type clip \
    --temperature 1 -et 2 -b 220 --lr 0.001 --ln_loss_lam 1 --margin_loss_lam 1 \
    --logit_scale_trainable True --lora_r 4 --lora_blocks 0 1 2 3 4 5 6 7 8 9 10 11 --use_margin True \
    --null_thres_mode adaptive --use_null_space --data_root "$DATA_ROOT" --seed 2024 "$@"
