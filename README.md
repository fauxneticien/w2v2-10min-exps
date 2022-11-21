## 1. Set up

### 1.1 Install Python package requirements

```bash
pip install -r requirements.txt
```

### 1.2 Install KenLM for language modelling

```bash
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
mkdir kenlm/build
cd kenlm/build
cmake ..
make -j2
```

Optional: move `bin/lmplz` and `bin/build_binary` to your `/usr/bin` directory. Otherwise replace calls to `lmplz` and `build_binary` with the correct paths on your machine.

## 2. Usage

**Demo files**. Aside from the raw data (e.g. ASR corpora/texts from the language documentation projects), various experiment artifacts have been archived on Zenodo (https://zenodo.org/record/7343344). We'll use some of these files to demo how to use the scripts in this repository.

### 2.1 wav2vec 2.0 fine-tuning

To fine-tune a wav2vec 2.0 model, use `train.py` with a configuration file (e.g. `--config=config.yaml`), whose parameters can be over-written by using dot notation for nested keys (e.g. `data.train_tsv='my_test.tsv'` to change the training data TSV file from the default `train.tsv`). See the comments in `config.yaml` for the various configurable arguments.

```bash
python train.py --config=config.yaml \
    data.base_path=data/gos-demo \
    trainargs.output_dir=gos-demo
```

### 2.2 Decoding

#### 2.2.1 Download and extract files for demos

```bash
# wav2vec 2.0 model fine-tuned on 10 mins of Gronings
wget https://zenodo.org/record/7343344/files/gos_wav2vec2-xls-r-300m-1e-5.tgz?download=1 -O tmp/gos_wav2vec2-xls-r-300m-1e-5.tgz
tar -xvzf tmp/gos_wav2vec2-xls-r-300m-1e-5.tgz -C tmp/

# Language model and lexicon derived from a 80,000 token corpus of Gronings
wget https://zenodo.org/record/7343344/files/GOS-3gram_80_000-seed1.tgz?download=1 -O tmp/GOS-3gram_80_000-seed1.tgz
tar -xvzf tmp/GOS-3gram_80_000-seed1.tgz -C tmp/
```

#### 2.2.2 Extract logits from dev/test sets using wav2vec 2.0 model checkpoint and save as pickle file

We extract acoustic model logits into a pkl file so we don't need to keep re-extracting over and over again during a hyperparameter search in beam search decoding.

```bash
# dev set logits
python extract-am-logits.py \
    refs_tsv=data/gos-demo/dev.tsv \
    processor_dir=tmp/gos_wav2vec2-xls-r-300m-1e-5 \
    checkpoint_dir=tmp/gos_wav2vec2-xls-r-300m-1e-5/checkpoint-6500 \
    logits_pkl=tmp/logits_gos-demo-dev.pkl

# test set logits
python extract-am-logits.py \
    refs_tsv=data/gos-demo/test.tsv \
    processor_dir=tmp/gos_wav2vec2-xls-r-300m-1e-5 \
    checkpoint_dir=tmp/gos_wav2vec2-xls-r-300m-1e-5/checkpoint-6500 \
    logits_pkl=tmp/logits_gos-demo-test.pkl
```

#### 2.2.3 Greedy decoding on test set

```bash
python decode_greedy.py \
    refs_tsv=data/gos-demo/test.tsv \
    processor_dir=tmp/gos_wav2vec2-xls-r-300m-1e-5 \
    logits_pkl=tmp/logits_gos-demo-test.pkl

# Output (WER,CER):
# 0.5316642120765832,0.1579697747362418
```

#### 2.2.4 Beam search decoding (with lexicon and language model) on test set

Using `lm_weight=2.77` and `word_score=1.21` based on a hyper-parameter search (see script below). Notice a 9.2\% absolute improvement compared to greedy decoding above (53.2% greedy vs. 44.0% beam search w/ lexicon & LM).

```bash
python decode_beam-search.py \
    refs_tsv=data/gos-demo/test.tsv \
    processor_dir=tmp/gos_wav2vec2-xls-r-300m-1e-5 \
    logits_pkl=tmp/logits_gos-demo-test.pkl \
    decoder.lexicon=tmp/GOS-3gram_80_000-seed1/lexicon.txt \
    decoder.lm=tmp/GOS-3gram_80_000-seed1/lm.bin \
    decoder.lm_weight=2.77 \
    decoder.word_score=1.21 \
    decoder.beam_size=1500

# Output (WER,CER)
# 0.44035346097201766,0.1497005988023952
```

## 3. Miscellaneous scripts

### 3.1 Run a hyper parameter search

To find optimal `lm_weight` (in the range [0, 5]) and `word_score` (in the range [-5, 5]) on your dev set and then use optimal parameters to calculate WER/CER on your test set.

```bash
# Output search log and results here
mkdir -p tmp/hp-search

python run_one-decode-exp.py \
    exp_dir=tmp/hp-search \
    dev_tsv=data/gos-demo/dev.tsv \
    test_tsv=data/gos-demo/test.tsv \
    dev_logits=tmp/logits_gos-demo-dev.pkl \
    test_logits=tmp/logits_gos-demo-test.pkl \
    processor_dir=tmp/gos_wav2vec2-xls-r-300m-1e-5 \
    decoder.lexicon=tmp/GOS-3gram_80_000-seed1/lexicon.txt \
    decoder.lm=tmp/GOS-3gram_80_000-seed1/lm.bin \
    search_iter=32

# Output (also saved to tmp/hp-search/results.csv)
#
# Hyperparameter search complete:
#
#     LM: tmp/GOS-3gram_80_000-seed1/lm.bin
#     Dev WER, CER:       0.3953488372093023, 0.1340007701193685,
#     Test WER, CER:      0.44035346097201766, 0.1497005988023952
#     Parameters: 
#         lm_weight: 2.77
#         word_score: 1.21
```

The bash script `run_all-decode-exps.sh` basically loops over 1) all datasets (e.g. English, Frisian, Gronings), 2) all corpora subset sizes (e.g. 8M, 80k, 8k tokens), 3) 5 seeds for each corpus size, running `run_one-decode-exp.py` for each configuration.

### 3.2 Derive language model and lexicon from text file

We don't have permission to release the Gronings corpus, so we'll use the open source Common Crawl corpus to demo how to derive a language model and lexicon, given a text file.

```bash
# Download Common Crawl EN corpus (distributed with TED-LIUM Release 2)
wget https://zenodo.org/record/7343344/files/commoncrawl-9pc.en?download=1 -O tmp/commoncrawl-9pc.en.txt

# Optional: sample sub-corpus from full corpus
python sample_corpus.py \
    corpus_txt='tmp/commoncrawl-9pc.en.txt' \
    output_txt='tmp/commoncrawl_80k.txt' \
    max_tokens=80_000 \
    random_seed=1

mkdir -p tmp/commoncrawl_80k_lm

# Generate lexicon
python generate_lexicon.py \
    --data tmp/commoncrawl_80k.txt \
    --outpath tmp/commoncrawl_80k_lm/

# Generate 3-gram langauge model arpa and convert to binary (latter loads faster)
# Assuming lmplz and build_binary are in your /usr/bin
lmplz -o 3 < tmp/commoncrawl_80k.txt > tmp/commoncrawl_80k_lm/lm.arpa
build_binary tmp/commoncrawl_80k_lm/lm.arpa tmp/commoncrawl_80k_lm/lm.bin
```

### 3.3 Determine best checkpoint

In our experiments we fine-tuned models for 12k steps, saving every 500 steps (e.g. 500, 1000, ..., 12000) so there were 24 checkpoints to choose from. The HuggingFace trainer saves them in folders such as `checkpoint-6500` and `checkpoint-12000` (to save space in the Zenodo archive we only uploaded the best and last checkpoint), to evaluate them on your dev set run:

```bash
python eval-all-cps.py \
    dataset_dir=data/gos-demo \
    dev_tsv=dev.tsv \
    lm_dir=tmp/GOS-3gram_80_000-seed1 \
    processor_dir=tmp/gos_wav2vec2-xls-r-300m-1e-5 \
    cps_glob='tmp/gos_wav2vec2-xls-r-300m-1e-5/checkpoint-*' \
    results_csv=tmp/gos_cp_eval.csv

# Output (also saved to tmp/gos_cp_eval.csv):

# Checkpoint with lowest WERs: 
#    Checkpoint  variable     value
# 1       12000  NoLM_WER  0.476744
# 3       12000  WiLM_WER  0.395349
```
