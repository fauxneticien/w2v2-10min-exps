import jiwer
import torch
import os
import soundfile as sf
import omegaconf as oc
import pandas as pd

from datasets import Dataset
from tqdm import tqdm

from helpers import (
    utils,
    w2v2
)

config = oc.OmegaConf.from_cli()

assert '--config' in config.keys(), """\n
    Please supply a base config file, e.g. 'python train.py --config=CONFIG_FILE.yml'.
    You can then over-ride config parameters, e.g. 'python train.py --config=CONFIG_FILE.yml trainargs.learning_rate=1e-5'
"""

config, wandb_run = utils.make_config(config)

utils.announce('Configuring model and reading data')

model, processor = w2v2.configure_hf_w2v2_model(config)
model = model.cuda()

devset_path = os.path.join(config['data'].base_path, config['data'].train_tsv)
testset_path = os.path.join(config['data'].base_path, config['data'].eval_tsv)

print(f"Development set: {devset_path}")
print(f"Test set: {testset_path}")

dev_ds = pd.read_csv(devset_path, sep = '\t')
test_ds = pd.read_csv(testset_path, sep = '\t')

def _read_audio(path):
    full_path = os.path.join(config['data'].base_path, path)

    data, sr = sf.read(full_path)

    assert sr == 16_000

    return data

dev_ds['audio'] = [ _read_audio(path) for path in tqdm(dev_ds['path'].to_list(), desc='Reading audio data') ]
test_ds['audio'] = [ _read_audio(path) for path in tqdm(test_ds['path'].to_list(), desc='Reading audio data') ]

if 'subset_train' in config['data'].keys():
    dev_ds = dev_ds.sample(frac=1, random_state=config['data']['subset_train']['seed']).copy().reset_index(drop=True)
    dev_ds = dev_ds[ dev_ds['audio'].apply(lambda s: len(s)/16_000).cumsum() <= (60 * config['data']['subset_train']['mins']) ].copy().reset_index(drop=True)

    print(f"Subsetted development data as specified: {config['data']['subset_train']['mins']} minutes, random seed {config['data']['subset_train']['seed']}. Rows kept: {len(dev_ds)}")

dev_ds = Dataset.from_pandas(dev_ds[['audio', 'text']])
test_ds = Dataset.from_pandas(test_ds[['audio', 'text']])

utils.announce('Evaluating model')

def evaluate(batch):
    inputs = processor(batch['audio'], sampling_rate=16_000, return_tensors='pt', padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to('cuda'), attention_mask=inputs.attention_mask.to('cuda')).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch['transcription'] = processor.batch_decode(pred_ids)
    
    return batch

dev_ds = dev_ds.map(evaluate, batched=True, batch_size=1)
test_ds = test_ds.map(evaluate, batched=True, batch_size=1)

wer_dev = round(jiwer.wer(dev_ds['text'], dev_ds['transcription']), 3)
cer_dev = round(jiwer.cer(dev_ds['text'], dev_ds['transcription']), 3)

wer_test = round(jiwer.wer(test_ds['text'], test_ds['transcription']), 3)
cer_test = round(jiwer.cer(test_ds['text'], test_ds['transcription']), 3)

utils.announce('Results on development data')

print(f"WER: {wer_dev} CER: {cer_dev}")

utils.announce('Results on test data')

print(f"WER: {wer_test} CER: {cer_test}")
