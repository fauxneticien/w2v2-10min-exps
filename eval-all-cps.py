import datasets as hfds
import glob
import jiwer
import numpy as np
import os
import pandas as pd
import soundfile as sf
import time
import torch
import transformers as hft

from natsort import natsorted
from torchaudio.models.decoder import ctc_decoder
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# Don't use more than 1 thread for Pytorch dataloader if you're
# going to use parallel processes as well
# https://github.com/pytorch/pytorch/issues/19996
torch.set_num_threads(1)

dataset_dir = "data/llight-1h_lspeech-other-dev-test"
dev_tsv = "dev-other.tsv"

lm_dir = "models/lm_librispeech-4gram_4GB"

processor_dir = "models/w2v2-large_llight10m0_2e-5_all-cps"
cps_glob      = "models/w2v2-large_llight10m0_2e-5_all-cps/checkpoint-*"

results_csv = "tmp/eng_all_cp_results.csv"

df = pd.read_csv(os.path.join(dataset_dir, dev_tsv), sep="\t")
df['audio'] = df.path.apply(lambda p: sf.read(os.path.join(dataset_dir, p))[0])

processor = hft.Wav2Vec2Processor.from_pretrained(processor_dir)

beam_search_decoder = ctc_decoder(
    lexicon=os.path.join(lm_dir, "lexicon.txt"),
    lm=os.path.join(lm_dir, "lm.bin"),
    tokens=list(processor.tokenizer.get_vocab().keys()),
    blank_token=processor.tokenizer.pad_token,
    sil_token=processor.tokenizer.word_delimiter_token,
    unk_word=processor.tokenizer.unk_token,
    nbest=1,
    beam_size=500,
    lm_weight=2,
    word_score=-1,
)

def logits_to_preds(logits):

    beam_search_result = beam_search_decoder(logits)
    beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
    
    return beam_search_transcript

all_cp_results = []

for cp_path in natsorted(glob.glob(cps_glob)):

    cp_name = os.path.basename(cp_path)

    model     = hft.Wav2Vec2ForCTC.from_pretrained(cp_path)
    model.eval()
    model.to('cuda')

    logits_list = []
    nolm_preds  = []

    for audio in tqdm(df.audio.to_list(), ncols=100, desc=f"Extracting logits using {cp_name}"):

        inputs   = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True).input_values

        with torch.no_grad():
            logits = model(inputs.to('cuda')).logits

        predicted_ids = torch.argmax(logits, dim=-1)

        # npy_path = 'w2v2-large_llight10m0_dev-other/' + os.path.basename(path).replace('.flac', '.npy')
        # np.save(npy_path, logits.cpu().detach().numpy())

        nolm_pred     = processor.batch_decode(predicted_ids)[0]
        nolm_preds.append(nolm_pred)

        logits_list.append(logits.cpu().detach())

    wilm_preds = process_map(
        logits_to_preds,
        logits_list,
        chunksize=1,
        ncols=100,
        desc = "Beam search decoding"
    )

    nolm_wer = jiwer.wer(df['text'].to_list(), nolm_preds)
    wilm_wer = jiwer.wer(df['text'].to_list(), wilm_preds)

    nolm_cer = jiwer.cer(df['text'].to_list(), nolm_preds)
    wilm_cer = jiwer.cer(df['text'].to_list(), wilm_preds)

    results = {
        "Checkpoint" : cp_name,
        "NoLM_WER" : nolm_wer,
        "NoLM_CER" : nolm_cer,
        "WiLM_WER" : wilm_wer,
        "WiLM_CER" : wilm_cer
    }

    all_cp_results.append(results)

    print(results)

all_df = pd.DataFrame(all_cp_results)
all_df.Checkpoint = all_df.Checkpoint.apply(lambda c: c.replace('checkpoint-', '')).astype(int)

all_df.to_csv(results_csv, index=False)

print("Checkpoint with lowest WERs: ")
wers_df = all_df[['Checkpoint', 'NoLM_WER', 'WiLM_WER']].melt(id_vars='Checkpoint')
print(wers_df.iloc[wers_df.groupby('variable')['value'].idxmin()])
