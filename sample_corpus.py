"""

This scripts takes an input corpus (corpus_txt), selects
lines randomly from the corpus (using random_seed) until
a maximum number of tokens is reached (max_tokens), then
writes the selected lines to an output file (output_txt). 

Example:

python sample_corpus.py \
    corpus_txt='my_big_corpus.txt' \
    output_txt='my_sampled.txt' \
    max_tokens=1000 \
    random_seed=1

"""

import mmap
import numpy as np
import omegaconf as oc

from tqdm import tqdm

# From https://blog.nelsonliu.me/2016/07/30/progress-bars-for-python-file-reading-with-tqdm/
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

config = oc.OmegaConf.from_cli()

total_lines = get_num_lines(config['corpus_txt'])

lines = []

print(f"Reading corpus from {config['corpus_txt']}")
with open(config['corpus_txt'], 'r') as itxt:
    for line in tqdm(itxt, total=total_lines, ncols=100):
        lines.append(line)

print(f"Generating random indices using seed {config['random_seed']}")
np.random.seed(config['random_seed'])
line_nums = np.random.choice(total_lines, total_lines, replace=False)

token_count   = 0
sampled_lines = []

print("Sampling lines ...")
for line_num in tqdm(line_nums, ncols=100):

    if token_count < int(config['max_tokens']):

        line_txt = lines[line_num]

        sampled_lines.append(line_txt)
        token_count += line_txt.count(" ") + 1

    else:

        break

print("Max tokens reached!")

print(f"Writing output to {config['output_txt']}")
with open(config['output_txt'], 'w') as otxt:
    otxt.writelines(sampled_lines)
