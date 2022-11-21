python decode_beam-search.py \
    logits_pkl=tmp/results/ENG/logits-dev.pkl \
    refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
    processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
    decoder.beam_size=1500 \
    decoder.lexicon=tmp/results/ENG/ENG-4gram_FULL/lexicon.txt

python decode_beam-search.py \
    logits_pkl=tmp/results/ENG/logits-test.pkl \
    refs_tsv=data/llight-1h_lspeech-other-dev-test/test-other.tsv \
    processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
    decoder.beam_size=1500 \
    decoder.lexicon=tmp/results/ENG/ENG-4gram_FULL/lexicon.txt

python decode_beam-search.py \
    logits_pkl=tmp/results/GOS/logits-dev.pkl \
    refs_tsv=data/20221103_gos/dev.tsv \
    processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
    decoder.beam_size=1500 \
    decoder.lexicon=tmp/results/GOS/GOS-4gram_FULL/lexicon.txt

python decode_beam-search.py \
    logits_pkl=tmp/results/GOS/logits-test.pkl \
    refs_tsv=data/20221103_gos/test.tsv \
    processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
    decoder.beam_size=1500 \
    decoder.lexicon=tmp/results/GOS/GOS-4gram_FULL/lexicon.txt

python decode_beam-search.py \
    logits_pkl=tmp/results/FRY/logits-dev.pkl \
    refs_tsv=data/20221103_fry/dev.tsv \
    processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
    decoder.beam_size=1500 \
    decoder.lexicon=tmp/results/FRY/FRY-4gram_FULL/lexicon.txt

python decode_beam-search.py \
    logits_pkl=tmp/results/FRY/logits-test.pkl \
    refs_tsv=data/20221103_fry/test.tsv \
    processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
    decoder.beam_size=1500 \
    decoder.lexicon=tmp/results/FRY/FRY-4gram_FULL/lexicon.txt

python extract-am-logits.py \
    refs_tsv=data/20221014_nasal/dev.tsv \
    processor_dir=models/nsy_wav2vec2-xls-r-300m-5e-5 \
    checkpoint_dir=models/nsy_wav2vec2-xls-r-300m-5e-5/checkpoint-8500/ \
    logits_pkl=tmp/logits_dev_nsy_wav2vec2-xls-r-300m-5e-5_8500.pkl

python extract-am-logits.py \
    refs_tsv=data/20221014_nasal/test.tsv \
    processor_dir=models/nsy_wav2vec2-xls-r-300m-5e-5 \
    checkpoint_dir=models/nsy_wav2vec2-xls-r-300m-5e-5/checkpoint-8500/ \
    logits_pkl=tmp/logits_test_nsy_wav2vec2-xls-r-300m-5e-5_8500.pkl

python extract-am-logits.py \
    refs_tsv=data/20221102_besemah/dev.tsv \
    processor_dir=models/pse_wav2vec2-xls-r-300m-5e-5 \
    checkpoint_dir=models/pse_wav2vec2-xls-r-300m-5e-5/checkpoint-11500/ \
    logits_pkl=tmp/logits_dev_pse_wav2vec2-xls-r-300m-5e-5_11500.pkl

python extract-am-logits.py \
    refs_tsv=data/20221102_besemah/test.tsv \
    processor_dir=models/pse_wav2vec2-xls-r-300m-5e-5 \
    checkpoint_dir=models/pse_wav2vec2-xls-r-300m-5e-5/checkpoint-11500/ \
    logits_pkl=tmp/logits_test_pse_wav2vec2-xls-r-300m-5e-5_11500.pkl

python decode_beam-search.py \
    logits_pkl=tmp/logits_dev_nsy_wav2vec2-xls-r-300m-5e-5_8500.pkl \
    refs_tsv=data/20221014_nasal/dev.tsv \
    processor_dir=models/nsy_wav2vec2-xls-r-300m-5e-5 \
    decoder.beam_size=1500 \
    decoder.lexicon=tmp/lexicon_nasal_9.5k.txt

python decode_beam-search.py \
    logits_pkl=tmp/logits_test_nsy_wav2vec2-xls-r-300m-5e-5_8500.pkl \
    refs_tsv=data/20221014_nasal/test.tsv \
    processor_dir=models/nsy_wav2vec2-xls-r-300m-5e-5 \
    decoder.beam_size=1500 \
    decoder.lexicon=tmp/lexicon_nasal_9.5k.txt

python decode_beam-search.py \
    logits_pkl=tmp/logits_dev_pse_wav2vec2-xls-r-300m-5e-5_11500.pkl \
    refs_tsv=data/20221102_besemah/dev.tsv \
    processor_dir=models/pse_wav2vec2-xls-r-300m-5e-5 \
    decoder.beam_size=1500 \
    decoder.lexicon=tmp/lexicon_besemah_9.5k.txt

python decode_beam-search.py \
    logits_pkl=tmp/logits_test_pse_wav2vec2-xls-r-300m-5e-5_11500.pkl \
    refs_tsv=data/20221102_besemah/test.tsv \
    processor_dir=models/pse_wav2vec2-xls-r-300m-5e-5 \
    decoder.beam_size=1500 \
    decoder.lexicon=tmp/lexicon_besemah_9.5k.txt
