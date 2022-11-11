import omegaconf as oc
import pandas as pd
import pickle
import soundfile as sf
import torch
import transformers as hft

from pathlib import Path
from tqdm import tqdm

config = oc.OmegaConf.from_cli()

print(f"Loading audio and text specified in {config.refs_tsv}")
refs_df   = pd.read_csv(config.refs_tsv, sep="\t")
base_path = Path(config.refs_tsv).parent

refs_df['audio'] = refs_df.path.apply(lambda p: sf.read(base_path / p)[0])

print(f"Loading processor from {config.processor_dir}")
processor = hft.Wav2Vec2Processor.from_pretrained(config.processor_dir)

print(f"Loading model checkpoint from {config.checkpoint_dir}")
model= hft.Wav2Vec2ForCTC.from_pretrained(config.checkpoint_dir)
model.eval()
model.to('cuda')

logits_list = []

for audio in tqdm(refs_df.audio.to_list(), ncols=100, desc=f"Extracting logits"):

    inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True).input_values

    with torch.no_grad():
        logits = model(inputs.to('cuda')).logits

    logits_list.append(logits.cpu().detach())

print(f"Writing output to {config.logits_pkl}")
with open(config.logits_pkl, 'wb') as handle:
    pickle.dump(logits_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

