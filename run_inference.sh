#!/bin/bash
datasets=("inquisitive" "reccon")
methods=("random" "edit_distance" "bm25" "semantic_similarity" "minilm" "cross_encoder" "rerank" "chatgpt" "gpt2" "gpt_j" "opt_6b")
score_methods=("single-sentence" "auto-regressive" "causal-full")
priority='standard'

for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        datasplit="test"
        fname="results/"${dataset}"/"${method}".txt"
        echo ${fname}
        if [ ${method} == "rerank" ] || [ ${method} == "chatgpt" ]; then
            nlprun -a backtracing -g 1 -d a6000 ' python3 scripts/evaluation/run_rerank_and_api.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit} -n ${method}_${dataset} -p ${priority}

        elif [ ${method} == "gpt2" ] || [ ${method} == "opt_6b" ] || [ ${method} == "gpt_j" ]; then
            for score_method in "${score_methods[@]}"; do
                fname="results/"${dataset}"/"${method}"_"${score_method}".txt"
                nlprun -a backtracing -g 1 -d a6000 'python3 scripts/evaluation/run_treatment.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit}'  --collect_logprobs --score_method='${score_method} -n ${method}_${dataset}_${score_method} -p ${priority} --mem 128G
            done

        else
            nlprun -a backtracing -g 1  -d a6000 'python3 scripts/evaluation/run.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit} -n ${method}_${dataset} -p ${priority}
        fi
    done
done

dataset="sight" # sight is the only dataset that runs on chunks
for method in "${methods[@]}"; do
    datasplit="annotated"
    fname="results/"${dataset}"/"${method}".txt"
    echo ${fname}
    if [ ${method} == "rerank" ] || [ ${method} == "chatgpt" ]; then
        nlprun -a backtracing -g 1 -d a6000 'python3 scripts/evaluation/run_rerank_and_api.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit} -n ${method}_${dataset} -p ${priority}
        # > ${fname}
    elif [ ${method} == "gpt2" ] || [ ${method} == "opt_6b" ] || [ ${method} == "gpt_j" ]; then
        # datasplit="test_chunks_window20"
        datasplit="annotated_chunks_window20"
        for score_method in "${score_methods[@]}"; do
            nlprun -a backtracing -g 1  -d a6000 'python3 scripts/evaluation/run_treatment.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit}' --collect_logprobs --score_method='${score_method} -n ${method}_${dataset}_${score_method} -p ${priority} --mem 128G
        done
    else
        nlprun -a backtracing  -d a6000 'python3 scripts/evaluation/run.py --dataset='${dataset}' --method='${method}' --datasplit='${datasplit}  -n ${method}_${dataset} -p ${priority}
    fi
done
