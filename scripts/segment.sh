
python train.py -c ./configs/db_mobilenet_v3_large_big.yml \
    --train-data /home/phung/AnhHung/data/bussiness_card/segment_label/index.txt \
    --val-data /home/phung/AnhHung/data/bussiness_card/segment_label/index.txt \
    --total-steps 10000 \
    --batch-size 4 \
    --num-workers 1 \
    --validate-every 250 \
    --learning-rate 3e-4 