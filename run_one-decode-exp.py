import omegaconf as oc
import os
import pandas as pd

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from helpers.utils import announce
from itertools import count
from subprocess import Popen, PIPE, CalledProcessError

config = oc.OmegaConf.from_cli()

# BayesianOptimization doesn't seem to pass in iteration so let's keep
# a global variable for counting
counter = count(start=1)

def lm_decode(lm_weight, word_score, decode_on='dev', beam_size=500, output='wer_inverse'):

    iteration = next(counter)

    if beam_size==500:
        # Print out weights if we're doing the search with beam size 500
        # For the final run, we'll use beam size 1500 as in the original wav2vec 2.0 paper
        print("--")
        print(f"{iteration=} of {config.search_iter}, {lm_weight=}, {word_score=}")

    assert decode_on in ['dev', 'test'], ValueError("Unrecognized value for 'decode_on' argument")

    cmd = [
        "python", "decode_beam-search.py",
        f"logits_pkl='{config.dev_logits}'" if decode_on == 'dev' else f"logits_pkl='{config.test_logits}'",
        f"refs_tsv='{config.dev_tsv}'"      if decode_on == 'dev' else f"refs_tsv='{config.test_tsv}'",
        f"processor_dir='{config.processor_dir}'",
        f"decoder.beam_size={beam_size}",
        f"decoder.lexicon='{config.decoder.lexicon}'",
        f"decoder.lm='{config.decoder.lm}'",
        f"decoder.lm_weight={lm_weight}",
        f"decoder.word_score={word_score}"
    ]

    with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            # Print stdout from subprocess as you go so
            # errors propagate back to the main process
            print(line, end='')

    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)

    # Important: we assume that the last thing decode_beam-search.py prints
    # are two numbers 'X, Y' where X = WER, Y = CER
    wer, cer = [ float(v) for v in line.split(",") ]

    if output=='wer_inverse':
        # Return 1 - wer to work with maximize() function in hyperparameter search
        return 1 - wer
    elif output=='wer_cer':
        # Return wer, cer for final decoding on dev and test after best hyperparameters found
        return [wer, cer]
    else:
        raise ValueError(f"Unrecognized value for 'output' argument")

announce('Start of experiment')

optimizer = BayesianOptimization(
    f=lm_decode,
    random_state=1,
    pbounds={
        'lm_weight': (0, 5),
        'word_score': (-5, 5)
        }
)

logger = JSONLogger(path=os.path.join(config.exp_dir, "logs.json"))
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(init_points=2, n_iter=config.search_iter)

print("Hyperparameter search complete. Getting final scores with best parameters ...")

best_dev_wer, best_dev_cer = lm_decode(
    optimizer.max['params']['lm_weight'],
    optimizer.max['params']['word_score'],
    decode_on='dev',
    beam_size=1500,
    output='wer_cer'
)

best_test_wer, best_test_cer = lm_decode(
    optimizer.max['params']['lm_weight'],
    optimizer.max['params']['word_score'],
    decode_on='test',
    beam_size=1500,
    output='wer_cer'
)

print(f"""
Hyperparameter search complete:

    LM: {config.decoder.lm}
    Dev WER, CER:\t{best_dev_wer}, {best_dev_cer},
    Test WER, CER:\t{best_test_wer}, {best_test_cer}
    Parameters: 
        lm_weight: {round(optimizer.max['params']['lm_weight'], 2)}
        word_score: {round(optimizer.max['params']['word_score'], 2)}
""")

pd.DataFrame({
    'lm_weight'  : [ optimizer.max['params']['lm_weight'] ],
    'word_score' : [ optimizer.max['params']['word_score'] ],
    'wer_dev' :    [ best_dev_wer ],
    'cer_dev' :    [ best_dev_cer ],
    'wer_test' :   [ best_test_wer ],
    'cer_test' :   [ best_test_cer ]
}).to_csv(os.path.join(config.exp_dir, "results.csv"), index=False)

announce('End of experiment')
