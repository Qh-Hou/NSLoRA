DATA_ROOT="${DOMAINNET_ROOT:-${DATA_ROOT:-DATA_ROOT}}"

python train_eval.py -d sdomainet \
    -b 256 --temperature 20 --lr 0.005 --ln_loss_lam 1 \
    --use_margin True --margin_loss_lam 0.001 \
    --lora_r 4 --lora_blocks 0 1 2 3 4 5 6 7 8 9 10 11 \
    --null_thres_mode adaptive --null_eta1 0.92 --use_null_space --data_root "$DATA_ROOT" --seed 2024 "$@"
