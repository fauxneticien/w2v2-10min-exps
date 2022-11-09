import pandas as pd
import transformers as hft
import os
import pickle
import soundfile as sf
import torch

from bayes_opt import BayesianOptimization
from tqdm import tqdm

import subprocess

base_path   = "data/20221103_fry"
dev_tsv     = "dev.tsv"
test_tsv    = "test.tsv"

dev_tsv     = os.path.join(base_path, dev_tsv)
test_tsv    = os.path.join(base_path, test_tsv)

lexicon_path = "models/lm_fry-2gram_1MB/lexicon.txt"
lm_path      = "models/lm_fry-2gram_1MB/lm.bin"

processor_dir = "models/fry_wav2vec2-xls-r-300m-1e-5"
cp_dir        = "models/fry_wav2vec2-xls-r-300m-1e-5/checkpoint-9000"

df = pd.read_csv(dev_tsv, sep="\t")

df['audio'] = df.path.apply(lambda p: sf.read(os.path.join(base_path, p))[0])

processor = hft.Wav2Vec2Processor.from_pretrained(processor_dir)

# model     = hft.Wav2Vec2ForCTC.from_pretrained(cp_dir)
# model.eval()
# model.to('cuda')

# logits_list = []

# for audio in tqdm(df.audio.to_list(), ncols=100, desc=f"Extracting logits"):

#     inputs   = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True).input_values

#     with torch.no_grad():
#         logits = model(inputs.to('cuda')).logits

#     logits_list.append(logits.cpu().detach())

# with open('tmp/logits.pickle', 'wb') as handle:
#     pickle.dump(logits_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def lm_decode(lm_weight, word_score):

    stdout = subprocess.check_output([
        "python", "lm-decode.py",
        f"data.refs_tsv='{dev_tsv}'",
        "logits_npy='tmp/logits.pickle'",
        f"processor_dir='{processor_dir}'",
        "decoder.beam_size=500",
        f"decoder.lexicon='{lexicon_path}'",
        f"decoder.lm='{lm_path}'",
        f"decoder.lm_weight={lm_weight}",
        f"decoder.word_score={word_score}"
    ])

    wer, cer = [ float(v) for v in stdout.decode('ascii').split(", ") ]

    print(f"WER: {wer}, CER: {cer}")

    # Return 1 - wer to work with maximize() function in
    # hyperparameter search
    return 1 - wer

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

pbounds = {
    'lm_weight': (0, 5),
    'word_score': (-5, 5)
}

optimizer = BayesianOptimization(
    f=lm_decode,
    pbounds=pbounds,
    random_state=1,
)

logger = JSONLogger(path="tmp/logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=2,
    n_iter=2,
)

# Re-decode dev set with beam size 1500
stdout = subprocess.check_output([
    "python", "lm-decode.py",
    f"data.refs_tsv='{dev_tsv}'",
    "logits_npy='tmp/logits.pickle'",
    f"processor_dir='{processor_dir}'",
    "decoder.beam_size=1500",
    f"decoder.lexicon='{lexicon_path}'",
    f"decoder.lm='{lm_path}'",
    f"decoder.lm_weight={optimizer.max['params']['lm_weight']}",
    f"decoder.word_score={optimizer.max['params']['word_score']}"
])

best_dev_wer, best_dev_cer = [ round(float(v)*100, 2) for v in stdout.decode('ascii').split(", ") ]

# Decode test set

print(f"""
Hyperparameter search complete:

    Lowest WER, dev: {best_dev_wer} (CER: {best_dev_cer})
    Parameters: 
        lm_weight: {round(optimizer.max['params']['lm_weight'], 2)}
        word_score: {round(optimizer.max['params']['word_score'], 2)}
""")
