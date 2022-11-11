import omegaconf as oc

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from subprocess import Popen, PIPE, CalledProcessError

config = oc.OmegaConf.from_cli()

def lm_decode(lm_weight, word_score):

    print("--")
    print(f"{lm_weight=}, {word_score=}")

    cmd = [
        "python", "decode_beam-search.py",
        f"logits_pkl='{config.logits_pkl}'",
        f"refs_tsv='{config.refs_tsv}'",
        f"processor_dir='{config.processor_dir}'",
        "decoder.beam_size=500",
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
    wer, cer = [ float(v) for v in line.split(", ") ]

    # Return 1 - wer to work with maximize() function in hyperparameter search
    return 1 - wer

optimizer = BayesianOptimization(
    f=lm_decode,
    random_state=1,
    pbounds={
        'lm_weight': (0, 5),
        'word_score': (-5, 5)
        }
)

logger = JSONLogger(path="tmp/logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(init_points=2, n_iter=128)

print(f"""
Hyperparameter search complete:

    LM: {config.decoder.lm}
    Lowest WER: {1 - optimizer.max['target']}
    Parameters: 
        lm_weight: {round(optimizer.max['params']['lm_weight'], 2)}
        word_score: {round(optimizer.max['params']['word_score'], 2)}
""")
