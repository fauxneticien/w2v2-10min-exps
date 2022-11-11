# Used to select the GPU that ISN'T driving the display on our local workbench
# uncomment for yourself if applicable
export CUDA_VISIBLE_DEVICES="1"

# Set parameters for each language

declare -rgA GOS=(
    # wav2vec 2.0 model checkpoint and processor paths
    [processor_dir]="models/gos_wav2vec2-xls-r-300m-1e-5"
    [checkpoint_dir]="models/gos_wav2vec2-xls-r-300m-1e-5/checkpoint-6500"

    [corpus_txt]="data/lm/gos_clean_40mb.txt"
    # Language model configs, space-separated "n_gram:sample_size" where
    # sample_size is indicated in number of tokens. This size is passed to
    # a Python script, which allows '_' in numbers (e.g. 800_000)
    [lm_configs]="3:80_000 2:8000"

    # Tune on LM alpha/beta on dev then evaluate on test with best parameters
    [dev_tsv]="data/20221103_gos/dev.tsv"
    [test_tsv]="data/20221103_gos/test.tsv"
)

declare -rgA FRY=(
    [processor_dir]="models/fry_wav2vec2-xls-r-300m-1e-5"
    [checkpoint_dir]="models/fry_wav2vec2-xls-r-300m-1e-5/checkpoint-9000"

    [corpus_txt]="data/lm/gos_clean_40mb.txt"
    # Language model configs, space-separated "n_gram:sample_size" where
    # sample_size is indicated in number of tokens. This size is passed to
    # a Python script, which allows '_' in numbers (e.g. 800_000)
    [lm_configs]="3:80_000 2:8000"

    [dev_tsv]="data/20221103_fry/dev.tsv"
    [test_tsv]="data/20221103_fry/test.tsv"
)

langs=("GOS" "FRY")

# Path to KenLM binaries
# See https://github.com/kpu/kenlm#compiling for installation instructions
LMPLZ_BIN='/usr/bin/lmplz'
BUILD_BIN='/usr/bin/build_binary'

OUTPUT_DIR="tmp/results"

for lang in "${langs[@]}"; do

    declare -n config="${lang}"

    # Set output directory and create (if necessary)
    LANG_DIR="tmp/results/${lang}"
    mkdir -p ${LANG_DIR}

    dev_logits="${LANG_DIR}/logits-dev.pkl"
    test_logits="${LANG_DIR}/logits-test.pkl"

    # Extract logits using wav2vec 2.0 model for LM decoding experiments

    From dev set (for hyperparameter tuning)
    python extract-am-logits.py \
        refs_tsv=${config[dev_tsv]} \
        processor_dir=${config[processor_dir]} \
        checkpoint_dir=${config[checkpoint_dir]} \
        logits_pkl=${dev_logits}

    From test set (for evaluation with best hyperparameter)
    python extract-am-logits.py \
        refs_tsv=${config[test_tsv]} \
        processor_dir=${config[processor_dir]} \
        checkpoint_dir=${config[checkpoint_dir]} \
        logits_pkl=${test_logits}

    # Run greedy decoding
    GREEDY_DIR="${LANG_DIR}/${lang}-greedy"
    mkdir -p ${GREEDY_DIR}

    # Decode on dev
    dev_wer_cer=$(python decode_greedy.py \
        logits_pkl="'${dev_logits}'" \
        refs_tsv="'${config[dev_tsv]}'" \
        processor_dir="'${config[processor_dir]}'")

    # Decode on test
    test_wer_cer=$(python decode_greedy.py \
        logits_pkl="'${test_logits}'" \
        refs_tsv="'${config[test_tsv]}'" \
        processor_dir="'${config[processor_dir]}'")

    # Write combined outputs to csv file (match the LM decoding experiments format below)
    echo "lm_weight,word_score,wer_dev,cer_dev,wer_test,cer_test" > "${GREEDY_DIR}/results.csv"
    echo "NA,NA,${dev_wer_cer},${dev_wer_cer}" >> "${GREEDY_DIR}/results.csv"

    for seed in {1..5}; do

        for lm_config in ${config[lm_configs]}; do

            IFS=':' read -r -a array <<< ${lm_config}

            n_gram=${array[0]}
            corpus_size=${array[1]}

            # Set output directory and create (if necessary)
            EXP_DIR="${LANG_DIR}/${lang}-${n_gram}gram_${corpus_size}-seed${seed}"
            mkdir -p ${EXP_DIR}

            # Configure file paths
            exp_corpus="${EXP_DIR}/corpus.txt"
            exp_lm_arpa="${EXP_DIR}/lm.arpa"
            exp_lm_bin="${EXP_DIR}/lm.bin"
            exp_lexicon="${EXP_DIR}/lexicon.txt"

            # Sample subset for experiment
            python sample_corpus.py \
                corpus_txt="'${config[corpus_txt]}'" \
                output_txt="'${exp_corpus}'" \
                max_tokens=${corpus_size} \
                random_seed=${seed}

            # Train language model
            $LMPLZ_BIN -o ${n_gram} < ${exp_corpus} > ${exp_lm_arpa}
            $BUILD_BIN ${exp_lm_arpa} ${exp_lm_bin}

            # Generate lexicon
            python generate_lexicon.py \
                --data ${exp_corpus} \
                --outpath "${EXP_DIR}/"

            python run_one-decode-exp.py \
                exp_dir="'${EXP_DIR}'"\
                dev_tsv="'${config[dev_tsv]}'" \
                test_tsv="'${config[test_tsv]}'" \
                dev_logits="'${dev_logits}'" \
                test_logits="'${test_logits}'" \
                processor_dir="'${config[processor_dir]}'" \
                decoder.lexicon="'${exp_lexicon}'" \
                decoder.lm="'${exp_lm_bin}'" \
                search_iter=32

        done

    done

done
