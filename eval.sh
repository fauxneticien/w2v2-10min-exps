#!/bin/bash

python eval.py --config=config.yaml trainargs.output_dir=test env.WANDB_MODE=offline \
    data.base_path=data/20221103_gos \
    data.train_tsv=dev.tsv \
    data.eval_tsv=test.tsv \
    data.subset_train.mins=10 \
    data.subset_train.seed=4892 \
    w2v2.model.pretrained_model_name_or_path=./checkpoints/gos/wav2vec2-xls-r-300m-1e-5/checkpoint-12000