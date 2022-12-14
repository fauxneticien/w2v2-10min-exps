import jiwer
import multiprocessing
import omegaconf as oc
import pandas as pd
import pickle
import torch
import transformers as hft
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
from torchaudio.models.decoder import ctc_decoder

# Ensure script can be imported by child processes using if __name__ == '__main__'
# https://superfastpython.com/multiprocessing-spawn-runtimeerror/
if __name__ == '__main__':

    # Don't use more than 1 thread for Pytorch dataloader if you're
    # going to use parallel processes as well
    # https://github.com/pytorch/pytorch/issues/19996
    torch.set_num_threads(1)

    config = oc.OmegaConf.from_cli()

    processor = hft.Wav2Vec2Processor.from_pretrained(config.processor_dir)

    # The _decoder function needs to be defined at the top level
    # (i.e. not nested inside a loop) to be pickle-able to take advantage of
    # parallel processing with process_map(), so we define decode_beam-search.py
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

    with open(config.logits_pkl, 'rb') as handle:
        logits_list = pickle.load(handle)

    with Pool(120) as p:
      preds = list(tqdm(p.imap(logits_to_preds, logits_list), ncols=100, total=len(logits_list)))

    refs_df  = pd.read_csv(config.refs_tsv, sep="\t")

    wer = jiwer.wer(refs_df['text'].to_list(), preds)
    cer = jiwer.cer(refs_df['text'].to_list(), preds)

    print(f"{wer},{cer}")
