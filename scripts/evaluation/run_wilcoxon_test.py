""" 
Run Wilcoxon signed-rank test on the results of the different models.

dataset="dailydialog"
for method in "${methods[@]}"; do
    datasplit="test"
    fname="results/"${dataset}"/"${method}".txt"
    echo ${fname}
    if [ ${method} == "rerank" ] || [ ${method} == "chatgpt" ]; then
        python3 scripts/evaluation/run_rerank_and_api.py --dataset=${dataset} --method=${method} --datasplit=${datasplit} --skip_checking > ${fname}
    elif [ ${method} == "gpt2" ] || [ ${method} == "opt_6b" ] || [ ${method} == "gpt_j" ]; then
        python3 scripts/evaluation/run_treatment.py --dataset=${dataset} --method=${method} --datasplit=${datasplit} --analyze_logprobs --skip_checking > ${fname}
    else
        python3 scripts/evaluation/run.py --dataset=${dataset} --method=${method} --datasplit=${datasplit} --skip_checking > ${fname}
    fi
    echo
done
methods = ("random" "edit_distance" "semantic_similarity" "minilm" "cross_encoder" "rerank" "chatgpt" "gpt2" "gpt_j" "opt_6b")
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.append(os.getcwd())

from scripts.evaluation.run_treatment import get_dirs_and_output_fname as get_treatment_dirs_and_output_fname
from scripts.evaluation.run_rerank_and_api import (
    get_dirs_and_output_fname as get_rerank_dirs_and_output_fname, 
    calculate_baseline_metrics as calculate_rerank_baseline_metrics
)
    
from scripts.evaluation.run import (
    get_dirs_and_output_fname, # as get_dirs_and_output_fname
    calculate_baseline_metrics
)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--method', type=str, default='random')
parser.add_argument('--datasplit', type=str, default='test')

args = parser.parse_args()

if __name__ == '__main__': 
    args.method='edit_distance'
    _, _, output_fname = get_dirs_and_output_fname(args)
    edit_df = pd.read_csv(output_fname, engine='python', encoding='utf-8', on_bad_lines='skip')
    edit_results_df = calculate_baseline_metrics(edit_df, )
    # Only use k = 1
    edit_results_df = edit_results_df[edit_results_df['top_k'] == 1]
    edit_results_df.sort_values(by=['fname', 'query_text'], inplace=True)
    # Get np array of correct, convert True to 1 and False to 0
    edit_correct = edit_results_df['correct'].to_numpy().astype(int)
    # scores_df = utils.get_scores_df(df, score_methods)
    # results_df = utils.get_results_df(scores_df, methods=score_methods, top_ks=top_ks)

    args.method='semantic_similarity'
    _, _, output_fname = get_dirs_and_output_fname(args)
    sem_df = pd.read_csv(output_fname, engine='python', encoding='utf-8', on_bad_lines='skip')  
    sem_results_df = calculate_baseline_metrics(sem_df)
    sem_results_df = sem_results_df[sem_results_df['top_k'] == 1]
    sem_results_df.sort_values(by=['fname', 'query_text'], inplace=True)
    # Get np array of correct
    sem_correct = sem_results_df['correct'].to_numpy().astype(int)

    args.method='rerank'
    _, _, output_fname = get_rerank_dirs_and_output_fname(args)
    rerank_df = pd.read_csv(output_fname, engine='python', encoding='utf-8', on_bad_lines='skip')
    rerank_results_df = calculate_rerank_baseline_metrics(rerank_df)
    rerank_results_df = rerank_results_df[rerank_results_df['top_k'] == 1]
    rerank_results_df.sort_values(by=['fname', 'query_text'], inplace=True)
    # Get np array of correct
    rerank_correct = rerank_results_df['correct'].to_numpy().astype(int)

    print(wilcoxon(edit_correct, sem_correct))
    
    import pdb; pdb.set_trace()