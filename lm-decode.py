import jiwer
import pandas as pd
import transformers as hft
import omegaconf as oc
import pickle

from tqdm.contrib.concurrent import process_map
from torchaudio.models.decoder import ctc_decoder

config = oc.OmegaConf.from_cli()

with open(config['logits_npy'], 'rb') as handle:
    logits_list = pickle.load(handle)

processor = hft.Wav2Vec2Processor.from_pretrained(config.processor_dir)

# The _decoder function needs to be defined at the top level
# (i.e. not nested inside a loop) to be pickle-able to take advantage of
# parallel processing with process_map(), so we define lm-decode.py
# as a standalone script which we call using subprocess.check_output()
# from another process
_decoder = ctc_decoder(
    tokens=list(processor.tokenizer.get_vocab().keys()),
    blank_token=processor.tokenizer.pad_token,
    sil_token=processor.tokenizer.word_delimiter_token,
    unk_word=processor.tokenizer.unk_token,
    nbest=1,
    **config['decoder']
)

def logits_to_preds(logits):

    beam_search_result = _decoder(logits)
    beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
    
    return beam_search_transcript

wilm_preds = process_map(
    logits_to_preds,
    logits_list,
    chunksize=1,
    ncols=100,
    desc = "Beam search decoding"
)

df  = pd.read_csv(config.data.refs_tsv, sep="\t")
wer = jiwer.wer(df['text'].to_list(), wilm_preds)
cer = jiwer.cer(df['text'].to_list(), wilm_preds)

print(f"{wer}, {cer}")
