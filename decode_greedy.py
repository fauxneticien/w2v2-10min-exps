import jiwer
import omegaconf as oc
import pandas as pd
import pickle
import torch
import transformers as hft

from multiprocessing import cpu_count
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from torchaudio.models.decoder import ctc_decoder

config = oc.OmegaConf.from_cli()

with open(config.logits_pkl, 'rb') as handle:
    logits_list = pickle.load(handle)

processor = hft.Wav2Vec2Processor.from_pretrained(config.processor_dir)

def greedy_decode(logits):

    predicted_ids  = torch.argmax(logits, dim=-1)
    # Subset [0] = first utterance from batch
    predicted_text = processor.batch_decode(predicted_ids)[0]

    return predicted_text

preds = [ greedy_decode(l) for l in tqdm(logits_list, ncols=100, desc="Running greedy decoding") ]

refs_df  = pd.read_csv(config.refs_tsv, sep="\t")

wer = jiwer.wer(refs_df['text'].to_list(), preds)
cer = jiwer.cer(refs_df['text'].to_list(), preds)

print(f"{wer}, {cer}")
