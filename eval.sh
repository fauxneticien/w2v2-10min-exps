#!/bin/bash

# gos 1e-5 wer 0.4064 cer 0.1037
echo "GRONINGS NO WARM START"
python eval.py --config=config.yaml trainargs.output_dir=test env.WANDB_MODE=offline \
    data.base_path=data/20221103_gos \
    data.train_tsv=dev.tsv \
    data.eval_tsv=test.tsv \
    data.subset_train.mins=10 \
    data.subset_train.seed=4892 \
    w2v2.model.pretrained_model_name_or_path=./checkpoints/gos/wav2vec2-xls-r-300m-1e-5/checkpoint-12000

# fry 1e-5 wer 0.5671 cer 0.1887
echo "FRISIAN NO WARM START"
python eval.py --config=config.yaml trainargs.output_dir=test env.WANDB_MODE=offline \
    data.base_path=data/20221103_fry \
    data.train_tsv=dev.tsv \
    data.eval_tsv=test.tsv \
    data.subset_train.mins=10 \
    data.subset_train.seed=4892 \
    w2v2.model.pretrained_model_name_or_path=./checkpoints/fry/wav2vec2-xls-r-300m-1e-5/checkpoint-12000

# gos 1e-5 wer 0.3389 cer 0.091
echo "GRONINGS WITH WARM START"
python eval.py --config=config.yaml trainargs.output_dir=test env.WANDB_MODE=offline \
    data.base_path=data/20221103_gos \
    data.train_tsv=dev.tsv \
    data.eval_tsv=test.tsv \
    data.subset_train.mins=10 \
    data.subset_train.seed=4892 \
    w2v2.model.pretrained_model_name_or_path=./checkpoints/nld-cgn-gos/wav2vec2-xls-r-300m-1e-5/checkpoint-12000

# fry 1e-5 wer 0.5249 cer 0.1778
echo "FRISIAN WITH WARM START"
python eval.py --config=config.yaml trainargs.output_dir=test env.WANDB_MODE=offline \
    data.base_path=data/20221103_fry \
    data.train_tsv=dev.tsv \
    data.eval_tsv=test.tsv \
    data.subset_train.mins=10 \
    data.subset_train.seed=4892 \
    w2v2.model.pretrained_model_name_or_path=./checkpoints/nld-cgn-fry/wav2vec2-xls-r-300m-1e-5/checkpoint-12000

# bes 5e-5 wer 0.6226 cer 0.2139
echo "BESEMAH NO WARM START"
python eval.py --config=config.yaml trainargs.output_dir=test env.WANDB_MODE=offline \
    data.base_path=data/20221102_besemah \
    data.train_tsv=dev.tsv \
    data.eval_tsv=test.tsv \
    data.subset_train.mins=10 \
    data.subset_train.seed=4892 \
    w2v2.model.pretrained_model_name_or_path=./checkpoints/besemah/wav2vec2-xls-r-300m-5e-5/checkpoint-12000

# nasal 5e-5 wer 0.6726 cer 0.2309
echo "NASAL NO WARM START"
python eval.py --config=config.yaml trainargs.output_dir=test env.WANDB_MODE=offline \
    data.base_path=data/20221014_nasal \
    data.train_tsv=dev.tsv \
    data.eval_tsv=test.tsv \
    data.subset_train.mins=10 \
    data.subset_train.seed=4892 \
    w2v2.model.pretrained_model_name_or_path=./checkpoints/nasal/wav2vec2-xls-r-300m-5e-5/checkpoint-12000

# bes 1e-5 wer 0.6108 cer 0.2060
echo "BESEMAH WITH WARM START"
python eval.py --config=config.yaml trainargs.output_dir=test env.WANDB_MODE=offline \
    data.base_path=data/20221102_besemah \
    data.train_tsv=dev.tsv \
    data.eval_tsv=test.tsv \
    data.subset_train.mins=10 \
    data.subset_train.seed=4892 \
    w2v2.model.pretrained_model_name_or_path=./checkpoints/indo-malay-besemah/wav2vec2-xls-r-300m-1e-5/checkpoint-12000

# nasal 5e-5 wer 0.6554 cer 0.2277
echo "NASAL WITH WARM START"
python eval.py --config=config.yaml trainargs.output_dir=test env.WANDB_MODE=offline \
    data.base_path=data/20221014_nasal \
    data.train_tsv=dev.tsv \
    data.eval_tsv=test.tsv \
    data.subset_train.mins=10 \
    data.subset_train.seed=4892 \
    w2v2.model.pretrained_model_name_or_path=./checkpoints/indo-malay-nasal/wav2vec2-xls-r-300m-5e-5/checkpoint-12000