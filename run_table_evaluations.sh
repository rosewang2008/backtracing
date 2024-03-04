#!/bin/bash
# Methods map to: Random, Edit, Bi-Encoder (Q&A), Bi-Encoder (all-MiniLM), Cross-Encoder, Re-Ranker, gpt-3.5-turbo-16k, and then the likelihood-based methods
methods=("random" "edit_distance" "bm25" "semantic_similarity" "minilm" "cross_encoder" "rerank" "chatgpt" "gpt2" "gpt_j" "opt_6b")
likelihood_methods=("single-sentence" "auto-regressive" "causal-full")

# zip folders -> unzip if their unzipped version does not exist
zip_folders=("results/sight/annotated.zip" "results/sight/annotated_chunks_window20.zip" "results/reccon/test.zip" "results/inquisitive/test.zip")
target_folder=("results/sight/" "results/sight/" "results/reccon/" "results/inquisitive/")
for i in "${!zip_folders[@]}"; do
    zip_folder=${zip_folders[$i]}
    target=${target_folder[$i]}
    if [ ! -d "${zip_folder%.zip}" ]; then
        echo "Unzipping ${zip_folder} into ${target}"
        unzip -o ${zip_folder} -d ${target}
    fi
done


#### Retrieving Cause of Question in Lectures  ####
dataset="sight" # sight is the only dataset that runs on chunks
for method in "${methods[@]}"; do
    datasplit="annotated"
    fname="results/"${dataset}"/"${method}".txt"
    if [ ${method} == "rerank" ] || [ ${method} == "chatgpt" ]; then
        # nlprun -a backtracing -g 1 -d a6000 'python3 scripts/evaluation/run_rerank_and_api.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit}'  --analyze  --skip_checking > '${fname} -n ${method}_${dataset}
        python3 scripts/evaluation/run_rerank_and_api.py --dataset=${dataset} --method=${method} --datasplit=${datasplit}  --analyze  --skip_checking > ${fname}

    elif [ ${method} == "gpt2" ] || [ ${method} == "opt_6b" ] || [ ${method} == "gpt_j" ]; then
        datasplit="annotated_chunks_window20"
        # Go through score_methods - for inquisitive, we run the score methods in parallel to save time.
        for score_method in "${likelihood_methods[@]}"; do
            fname="results/"${dataset}"/"${method}"_"${score_method}".txt"
            # nlprun -a backtracing -g 1 -d a6000 'python3 scripts/evaluation/run_chunk_analysis.py --fname="results/sight/annotated_chunks_window20/"'${method}'"_"'${score_method}'".csv  > '${fname} -n ${method}_${dataset}
            python3 scripts/evaluation/run_chunk_analysis.py --fname=results/sight/annotated_chunks_window20/${method}_${score_method}.csv > ${fname}
        done

    else
        # nlprun -a backtracing -g 1 -d a6000 'python3 scripts/evaluation/run.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit}' --skip_checking > '${fname} -n ${method}_${dataset}
        python3 scripts/evaluation/run.py --dataset=${dataset} --method=${method} --datasplit=${datasplit} --skip_checking > ${fname}
    fi
    echo ${fname}
    echo
done

#### Retrieving the Cause of Question in News Articles ####
dataset="inquisitive"
for method in "${methods[@]}"; do
    datasplit="test"
    fname="results/"${dataset}"/"${method}".txt"
    if [ ${method} == "rerank" ] || [ ${method} == "chatgpt" ]; then
        # nlprun -a backtracing -g 1 -d a6000 'python3 scripts/evaluation/run_rerank_and_api.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit}'  --analyze --skip_checking  > '${fname} -n  ${method}_${dataset}
        python3 scripts/evaluation/run_rerank_and_api.py --dataset=${dataset} --method=${method} --datasplit=${datasplit}  --analyze --skip_checking  > ${fname}

    elif [ ${method} == "gpt2" ] || [ ${method} == "opt_6b" ] || [ ${method} == "gpt_j" ]; then
        # Go through score_methods - for inquisitive, we run the score methods in parallel to save time.
        for score_method in "${likelihood_methods[@]}"; do
            fname="results/"${dataset}"/"${method}"_"${score_method}".txt"
            # nlprun -a backtracing -g 1 -d a6000 'python3 scripts/evaluation/run_treatment.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit}'  --analyze_logprobs --skip_checking --score_method='${score_method}' > '${fname} -n ${method}_${dataset}
            python3 scripts/evaluation/run_treatment.py --dataset=${dataset} --method=${method} --datasplit=${datasplit}  --analyze_logprobs --skip_checking --score_method=${score_method} > ${fname}

        done

    else
        # nlprun -a backtracing -g 1 -d a6000 'python3 scripts/evaluation/run.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit}'  --skip_checking > '${fname}  -n  ${method}_${dataset}
        python3 scripts/evaluation/run.py --dataset=${dataset} --method=${method} --datasplit=${datasplit}  --skip_checking > ${fname}
    fi
    echo ${fname}
    echo
done


#### Retrieving the Cause of Emotion in Conversations ####
dataset="reccon"
for method in "${methods[@]}"; do
    datasplit="test"
    fname="results/"${dataset}"/"${method}".txt"
    echo ${fname}
    if [ ${method} == "rerank" ] || [ ${method} == "chatgpt" ]; then
        # nlprun -a backtracing -g 1 -d a6000 'python3 scripts/evaluation/run_rerank_and_api.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit}' --skip_checking --analyze > '${fname}  -n ${method}_${dataset}
        python3 scripts/evaluation/run_rerank_and_api.py --dataset=${dataset} --method=${method} --datasplit=${datasplit} --skip_checking --analyze > ${fname}

    elif [ ${method} == "gpt2" ] || [ ${method} == "opt_6b" ] || [ ${method} == "gpt_j" ]; then
        for score_method in "${likelihood_methods[@]}"; do
            fname="results/"${dataset}"/"${method}"_"${score_method}".txt"
            # nlprun -a backtracing -g 1 -d a6000 'python3 scripts/evaluation/run_treatment.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit}'  --analyze_logprobs --skip_checking --score_method='${score_method}'  > '${fname}  -n ${method}_${dataset}
            python3 scripts/evaluation/run_treatment.py --dataset=${dataset} --method=${method} --datasplit=${datasplit}  --analyze_logprobs --skip_checking --score_method=${score_method}  > ${fname}
        done

    else
        # nlprun -a backtracing -g 1 -d a6000 'python3 scripts/evaluation/run.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit}'  --skip_checking > '${fname}  -n ${method}_${dataset}
        python3 scripts/evaluation/run.py --dataset=${dataset} --method=${method} --datasplit=${datasplit}  --skip_checking > ${fname}
    fi
    echo
done
