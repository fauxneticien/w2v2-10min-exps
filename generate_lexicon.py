import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, default='../data/librispeech-lm-norm.txt', help="Path to the text file for LM training.")
    parser.add_argument("--outpath", default='../lexicon/', help="Path to the directory, where the lexicon is saved.")

    return parser.parse_args()

def read_data(pt):
    """Reads and normalizes text data"""

    # assert pt.endswith('.txt'), 'use a .txt file'
    f = open(pt, 'r').read().splitlines()
    
    words = [w.lower().split() for w in f]

    return sorted(list(set([i for w in words for i in w])))

def create_lex(words, outpath):
    """Create a lexicon"""
    
    os.makedirs(outpath, exist_ok=True)

    with open(outpath + 'lexicon.txt', 'w') as t:
        for word in words:
            chars = [c for c in word]
            t.write(word + '\t' + ' '.join(chars) +  ' |' + '\n')

def _main():
    args = parse_args()
    words = read_data(args.data)
    create_lex(words, args.outpath)

if __name__ == "__main__":
    _main()
