# Used to select GPU not driving the display on our local workbench
# uncomment if applicable
export CUDA_VISIBLE_DEVICES="1"

################# Decoding with official checkpoints/language models ###############

# On dev-other

# python extract-am-logits.py \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/wav2vec_big_10m_hf \
#     checkpoint_dir=models/wav2vec_big_10m_hf \
#     logits_pkl=tmp/logits_dev_wav2vec_big_10m_hf.pkl

## Greedy decode

# python decode.py \
#     logits_pkl=tmp/logits_dev_wav2vec_big_10m_hf.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/wav2vec_big_10m_hf \
#     method=greedy

## Decode with official 4-gram LM

# python decode.py \
#     logits_pkl=tmp/logits_dev_wav2vec_big_10m_hf.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/wav2vec_big_10m_hf \
#     method=beam_search \
#     decoder.beam_size=1500 \
#     decoder.lm=models/lm_librispeech-4gram_4GB-official/lm.bin \
#     decoder.lexicon=models/lm_librispeech-4gram_4GB-official/lexicon.txt \
#     decoder.lm_weight=3.23 \
#     decoder.word_score=-0.23

## Decode with re-trained 4-gram LM

# python decode.py \
#     logits_pkl=tmp/logits_dev_wav2vec_big_10m_hf.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/wav2vec_big_10m_hf \
#     method=beam_search \
#     decoder.beam_size=1500 \
#     decoder.lm=models/lm_librispeech-4gram_4GB/lm.bin \
#     decoder.lexicon=models/lm_librispeech-4gram_4GB/lexicon.txt \
#     decoder.lm_weight=3.23 \
#     decoder.word_score=-0.23

# On test-other

# python extract-am-logits.py \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/test-other.tsv \
#     processor_dir=models/wav2vec_big_10m_hf \
#     checkpoint_dir=models/wav2vec_big_10m_hf \
#     logits_pkl=tmp/logits_test_wav2vec_big_10m_hf.pkl

## Greedy decode

# python decode.py \
#     logits_pkl=tmp/logits_test_wav2vec_big_10m_hf.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/test-other.tsv \
#     processor_dir=models/wav2vec_big_10m_hf \
#     method=greedy

## Decode with official 4-gram LM

# python decode.py \
#     logits_pkl=tmp/logits_test_wav2vec_big_10m_hf.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/test-other.tsv \
#     processor_dir=models/wav2vec_big_10m_hf \
#     method=beam_search \
#     decoder.beam_size=1500 \
#     decoder.lm=models/lm_librispeech-4gram_4GB-official/lm.bin \
#     decoder.lexicon=models/lm_librispeech-4gram_4GB-official/lexicon.txt \
#     decoder.lm_weight=3.23 \
#     decoder.word_score=-0.23

## Decode with re-trained 4-gram LM

# python decode.py \
#     logits_pkl=tmp/logits_test_wav2vec_big_10m_hf.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/test-other.tsv \
#     processor_dir=models/wav2vec_big_10m_hf \
#     method=beam_search \
#     decoder.beam_size=1500 \
#     decoder.lm=models/lm_librispeech-4gram_4GB/lm.bin \
#     decoder.lexicon=models/lm_librispeech-4gram_4GB/lexicon.txt \
#     decoder.lm_weight=3.23 \
#     decoder.word_score=-0.23

################# Decoding with re-trained checkpoint and various language models ###############

## On dev-other

# python extract-am-logits.py \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     checkpoint_dir=models/w2v2-large_llight10m0_2e-5_all-cps/checkpoint-1500 \
#     logits_pkl=tmp/logits_dev_w2v2-large_llight10m0_2e-5_all-cps.pkl

# python decode.py \
#     logits_pkl=tmp/logits_dev_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     method=greedy

# python decode.py \
#     logits_pkl=tmp/logits_dev_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     method=beam_search \
#     decoder.beam_size=1500 \
#     decoder.lm=models/lm_librispeech-4gram_4GB/lm.bin \
#     decoder.lexicon=models/lm_librispeech-4gram_4GB/lexicon.txt \
#     decoder.lm_weight=4.996350805489099 \
#     decoder.word_score=-1.7929971848878337

# python hparam-search.py \
#     logits_pkl=tmp/logits_dev_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     decoder.lexicon=models/lm_librispeech-4gram_40MB/lexicon.txt \
#     decoder.lm=models/lm_librispeech-4gram_40MB/lm.bin

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_dev_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     decoder.lexicon=models/lm_librispeech-4gram_40MB/lexicon.txt \
#     decoder.lm=models/lm_librispeech-4gram_40MB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=1.74 \
#     decoder.word_score=2.86

# python hparam-search.py \
#     logits_pkl=tmp/logits_dev_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     decoder.lexicon=models/lm_librispeech-4gram_40MB-v2/lexicon.txt \
#     decoder.lm=models/lm_librispeech-4gram_40MB-v2/lm.bin

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_dev_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     decoder.lexicon=models/lm_librispeech-4gram_40MB-v2/lexicon.txt \
#     decoder.lm=models/lm_librispeech-4gram_40MB-v2/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=4.59 \
#     decoder.word_score=-2.79

# python hparam-search.py \
#     logits_pkl=tmp/logits_dev_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     decoder.lexicon=models/lm_librispeech-4gram_4MB/lexicon.txt \
#     decoder.lm=models/lm_librispeech-4gram_4MB/lm.bin

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_dev_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     decoder.lexicon=models/lm_librispeech-4gram_4MB/lexicon.txt \
#     decoder.lm=models/lm_librispeech-4gram_4MB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=1.89 \
#     decoder.word_score=2.05

# python hparam-search.py \
#     logits_pkl=tmp/logits_dev_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     decoder.lexicon=models/lm_librispeech-4gram_400KB/lexicon.txt \
#     decoder.lm=models/lm_librispeech-4gram_400KB/lm.bin

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_dev_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/dev-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     decoder.lexicon=models/lm_librispeech-4gram_400KB/lexicon.txt \
#     decoder.lm=models/lm_librispeech-4gram_400KB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=2.61 \
#     decoder.word_score=0.61

# On test-other

# python extract-am-logits.py \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/test-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     checkpoint_dir=models/w2v2-large_llight10m0_2e-5_all-cps/checkpoint-1500 \
#     logits_pkl=tmp/logits_test_w2v2-large_llight10m0_2e-5_all-cps.pkl

# python decode.py \
#     logits_pkl=tmp/logits_test_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/test-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     method=greedy

# python decode.py \
#     logits_pkl=tmp/logits_test_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/test-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     method=beam_search \
#     decoder.beam_size=1500 \
#     decoder.lm=models/lm_librispeech-4gram_4GB/lm.bin \
#     decoder.lexicon=models/lm_librispeech-4gram_4GB/lexicon.txt \
#     decoder.lm_weight=4.996350805489099 \
#     decoder.word_score=-1.7929971848878337

# python decode.py \
    # logits_pkl=tmp/logits_test_w2v2-large_llight10m0_2e-5_all-cps.pkl \
    # refs_tsv=data/llight-1h_lspeech-other-dev-test/test-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     method=beam_search \
#     decoder.beam_size=1500 \
#     decoder.lm=models/lm_librispeech-4gram_40MB/lm.bin \
#     decoder.lexicon=models/lm_librispeech-4gram_40MB/lexicon.txt \
#     decoder.lm_weight=1.7421087278347747 \
#     decoder.word_score=2.8554633398637375

python decode_beam-search.py \
    logits_pkl=tmp/logits_test_w2v2-large_llight10m0_2e-5_all-cps.pkl \
    refs_tsv=data/llight-1h_lspeech-other-dev-test/test-other.tsv \
    processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
    decoder.lexicon=models/lm_librispeech-4gram_40MB-v2/lexicon.txt \
    decoder.lm=models/lm_librispeech-4gram_40MB-v2/lm.bin \
    decoder.beam_size=1500 \
    decoder.lm_weight=4.59 \
    decoder.word_score=-2.79

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_test_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/test-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     decoder.lexicon=models/lm_librispeech-4gram_4MB/lexicon.txt \
#     decoder.lm=models/lm_librispeech-4gram_4MB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=1.89 \
#     decoder.word_score=2.05

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_test_w2v2-large_llight10m0_2e-5_all-cps.pkl \
#     refs_tsv=data/llight-1h_lspeech-other-dev-test/test-other.tsv \
#     processor_dir=models/w2v2-large_llight10m0_2e-5_all-cps \
#     decoder.lexicon=models/lm_librispeech-4gram_400KB/lexicon.txt \
#     decoder.lm=models/lm_librispeech-4gram_400KB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=2.61 \
#     decoder.word_score=0.61

################# Gronings ###############

## On dev

# python extract-am-logits.py \
#     refs_tsv=data/20221103_gos/dev.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     checkpoint_dir=models/gos_wav2vec2-xls-r-300m-1e-5/checkpoint-6500 \
#     logits_pkl=tmp/logits_dev_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl

# python decode_greedy.py \
#     logits_pkl=tmp/logits_dev_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
#     refs_tsv=data/20221103_gos/dev.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5

# python hparam-search.py \
#     logits_pkl=tmp/logits_dev_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
#     refs_tsv=data/20221103_gos/dev.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_gos-4gram_40MB/lexicon.txt \
#     decoder.lm=models/lm_gos-4gram_40MB/lm.bin

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_dev_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
#     refs_tsv=data/20221103_gos/dev.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_gos-4gram_40MB/lexicon.txt \
#     decoder.lm=models/lm_gos-4gram_40MB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=3.39 \
#     decoder.word_score=-1.29

# python hparam-search.py \
#     logits_pkl=tmp/logits_dev_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
#     refs_tsv=data/20221103_gos/dev.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_gos-3gram_4MB/lexicon.txt \
#     decoder.lm=models/lm_gos-3gram_4MB/lm.bin

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_dev_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
#     refs_tsv=data/20221103_gos/dev.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_gos-3gram_4MB/lexicon.txt \
#     decoder.lm=models/lm_gos-3gram_4MB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=3.3 \
#     decoder.word_score=-1.28

# python hparam-search.py \
#     logits_pkl=tmp/logits_dev_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
#     refs_tsv=data/20221103_gos/dev.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_gos-2gram_4KB/lexicon.txt \
#     decoder.lm=models/lm_gos-2gram_4KB/lm.bin

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_dev_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
#     refs_tsv=data/20221103_gos/dev.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_gos-2gram_4KB/lexicon.txt \
#     decoder.lm=models/lm_gos-2gram_4KB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=3.5 \
#     decoder.word_score=-1.7

# python hparam-search.py \
#     logits_pkl=tmp/logits_dev_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
#     refs_tsv=data/20221103_gos/dev.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_gos-2gram_40KB/lexicon.txt \
#     decoder.lm=models/lm_gos-2gram_40KB/lm.bin

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_dev_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
#     refs_tsv=data/20221103_gos/dev.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_gos-2gram_40KB/lexicon.txt \
#     decoder.lm=models/lm_gos-2gram_40KB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=4.34 \
#     decoder.word_score=-0.17

## On test

# python extract-am-logits.py \
#     refs_tsv=data/20221103_gos/test.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     checkpoint_dir=models/gos_wav2vec2-xls-r-300m-1e-5/checkpoint-6500 \
#     logits_pkl=tmp/logits_test_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl

# python decode_greedy.py \
#     logits_pkl=tmp/logits_test_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
#     refs_tsv=data/20221103_gos/test.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_test_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
#     refs_tsv=data/20221103_gos/test.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_gos-4gram_40MB/lexicon.txt \
#     decoder.lm=models/lm_gos-4gram_40MB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=3.39 \
#     decoder.word_score=-1.29

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_test_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
#     refs_tsv=data/20221103_gos/test.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_gos-3gram_4MB/lexicon.txt \
#     decoder.lm=models/lm_gos-3gram_4MB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=3.3 \
#     decoder.word_score=-1.28

# python decode_beam-search.py \
    # logits_pkl=tmp/logits_test_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
    # refs_tsv=data/20221103_gos/test.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_gos-2gram_4KB/lexicon.txt \
#     decoder.lm=models/lm_gos-2gram_4KB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=3.5 \
#     decoder.word_score=-1.7

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_test_gos_wav2vec2-xls-r-300m-1e-5_6500.pkl \
#     refs_tsv=data/20221103_gos/test.tsv \
#     processor_dir=models/gos_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_gos-2gram_40KB/lexicon.txt \
#     decoder.lm=models/lm_gos-2gram_40KB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=4.34 \
#     decoder.word_score=-0.17

################# Frisian ###############

## On dev

# python extract-am-logits.py \
#     refs_tsv=data/20221103_fry/dev.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
#     checkpoint_dir=models/fry_wav2vec2-xls-r-300m-1e-5/checkpoint-9000 \
#     logits_pkl=tmp/logits_dev_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl

# python decode_greedy.py \
#     logits_pkl=tmp/logits_dev_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
#     refs_tsv=data/20221103_fry/dev.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5

# python hparam-search.py \
#     logits_pkl=tmp/logits_dev_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
#     refs_tsv=data/20221103_fry/dev.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_fry-4gram_40MB/lexicon.txt \
#     decoder.lm=models/lm_fry-4gram_40MB/lm.bin

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_dev_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
#     refs_tsv=data/20221103_fry/dev.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_fry-4gram_40MB/lexicon.txt \
#     decoder.lm=models/lm_fry-4gram_40MB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=3.16 \
#     decoder.word_score=-1.29

# python hparam-search.py \
#     logits_pkl=tmp/logits_dev_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
#     refs_tsv=data/20221103_fry/dev.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_fry-3gram_4MB/lexicon.txt \
#     decoder.lm=models/lm_fry-3gram_4MB/lm.bin

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_dev_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
#     refs_tsv=data/20221103_fry/dev.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_fry-3gram_4MB/lexicon.txt \
#     decoder.lm=models/lm_fry-3gram_4MB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=2.9 \
#     decoder.word_score=-2.22

# python hparam-search.py \
#     logits_pkl=tmp/logits_dev_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
#     refs_tsv=data/20221103_fry/dev.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_fry-2gram_4KB/lexicon.txt \
#     decoder.lm=models/lm_fry-2gram_4KB/lm.bin

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_dev_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
#     refs_tsv=data/20221103_fry/dev.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_fry-2gram_4KB/lexicon.txt \
#     decoder.lm=models/lm_fry-2gram_4KB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=3.22 \
#     decoder.word_score=-1.61

# python hparam-search.py \
#     logits_pkl=tmp/logits_dev_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
#     refs_tsv=data/20221103_fry/dev.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
    # decoder.lexicon=models/lm_fry-1gram_4KB/lexicon.txt \
    # decoder.lm=models/lm_fry-1gram_4KB/lm.arpa

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_dev_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
#     refs_tsv=data/20221103_fry/dev.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_fry-1gram_4KB/lexicon.txt \
#     decoder.lm=models/lm_fry-1gram_4KB/lm.arpa \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=3.27 \
#     decoder.word_score=-3.36

## On test

# python extract-am-logits.py \
#     refs_tsv=data/20221103_fry/test.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
#     checkpoint_dir=models/fry_wav2vec2-xls-r-300m-1e-5/checkpoint-9000 \
#     logits_pkl=tmp/logits_test_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl

# python decode_greedy.py \
    # logits_pkl=tmp/logits_test_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
    # refs_tsv=data/20221103_fry/test.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_test_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
#     refs_tsv=data/20221103_fry/test.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_fry-4gram_40MB/lexicon.txt \
#     decoder.lm=models/lm_fry-4gram_40MB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=3.16 \
#     decoder.word_score=-1.29

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_test_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
#     refs_tsv=data/20221103_fry/test.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_fry-3gram_4MB/lexicon.txt \
#     decoder.lm=models/lm_fry-3gram_4MB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=2.9 \
#     decoder.word_score=-2.22

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_test_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
#     refs_tsv=data/20221103_fry/test.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_fry-2gram_4KB/lexicon.txt \
#     decoder.lm=models/lm_fry-2gram_4KB/lm.bin \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=3.22 \
#     decoder.word_score=-1.61

# python decode_beam-search.py \
#     logits_pkl=tmp/logits_test_fry_wav2vec2-xls-r-300m-1e-5_9000.pkl \
#     refs_tsv=data/20221103_fry/test.tsv \
#     processor_dir=models/fry_wav2vec2-xls-r-300m-1e-5 \
#     decoder.lexicon=models/lm_fry-1gram_4KB/lexicon.txt \
#     decoder.lm=models/lm_fry-1gram_4KB/lm.arpa \
#     decoder.beam_size=1500 \
#     decoder.lm_weight=3.27 \
#     decoder.word_score=-3.36
