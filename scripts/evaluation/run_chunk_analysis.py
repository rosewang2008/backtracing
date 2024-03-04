"""
cross_encoder: fname,query,query_idx,source,source_idx,query_text,target_idxs,prediction

LMs: fname,query,query_idx,source,source_idx,target_idxs,is_target,prefix_text,suffix_text,t,T,separator,prefix_length,suffix_length,complete_text_length,prefix_shorthand,suffix_shorthand,shorthand,complete_text,loss,logprob,normalized_logprob


fnames = cxTmmasBiC8_1_chunk11.json
<video_id>_<comment_id>_chunk<chunk_number>.json

No chunk score:
- For every video_id:
    - For every comment_id:
        - Get the top source_idx (highest prediction) from the cross_encoder.csv
        - Check if accurate

Chunk score:
- For every video_id:
    - For every comment_id:
        - Compute the average prediction score for each chunk
        - Take the top chunk and get the top source_idx (highest prediction) from the cross_encoder.csv
        - Check if accurate
"""

import pandas as pd
import seaborn as sns
import os
import sys
import json
sys.path.append(os.getcwd())
from scripts.evaluation import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str, default='')

args = parser.parse_args()

fname = args.fname

print(f"File: {fname}")

TOP_KS = [1, 3]

def load_df_and_score_keys(fname):
    df = pd.read_csv(fname)
    score_keys = ['prediction']
    # if 'gpt2', 'opt' or 'gpt_j' in fname, then run utils.get_scores_df
    if 'gpt2' in fname or 'opt' in fname or 'gpt_j' in fname:
        score_method = fname.split('/')[-1].split('_')[-1].split('.')[0]
        df = utils.get_scores_df(df, methods=[score_method])
        score_keys = [score_method]
    return df, score_keys

df, score_keys = load_df_and_score_keys(fname)

def get_video_ids(df):
    video_ids = df['fname'].apply(lambda x: x.split('_')[0])
    return video_ids.unique()

def get_comment_ids(df, video_id):
    video_df = df[df['fname'].apply(lambda x: x.split('_')[0]) == video_id]
    comment_ids = video_df['fname'].apply(lambda x: x.split('_')[1])
    return comment_ids.unique()

def get_chunk_ids(df, video_id, comment_id):
    video_df = df[df['fname'].apply(lambda x: x.split('_')[0]) == video_id]
    comment_df = video_df[video_df['fname'].apply(lambda x: x.split('_')[1]) == comment_id]
    chunk_ids = comment_df['fname'].apply(lambda x: x.split('_')[2].split('.')[0])
    return chunk_ids.unique()

def json_load_str_list(str_x):
    return json.loads(str_x.replace("'", '"'))

def get_top_chunk_df(video_comment_df, video_id, comment_id, score_key='prediction'):
    # Compute scores per chunk
    chunk_ids = get_chunk_ids(df, video_id, comment_id)
    chunk_scores = []
    for chunk_id in chunk_ids:
        chunk_df = video_comment_df[video_comment_df['fname'].apply(lambda x: x.split('_')[2].split('.')[0]) == chunk_id]
        chunk_scores.append(chunk_df[score_key].mean())

    # Get top chunk
    top_chunk_id = chunk_ids[chunk_scores.index(max(chunk_scores))]
    top_chunk_df = video_comment_df[video_comment_df['fname'].apply(lambda x: x.split('_')[2].split('.')[0]) == top_chunk_id]
    return top_chunk_df

def run_score(df, with_chunk_score, score_key='prediction'):
    results = []
    video_ids = get_video_ids(df)
    for video_id in video_ids:
        comment_ids = get_comment_ids(df, video_id)
        for comment_id in comment_ids:
            video_df = df[df['fname'].apply(lambda x: x.split('_')[0]) == video_id]
            comment_df = video_df[video_df['fname'].apply(lambda x: x.split('_')[1]) == comment_id]
            if with_chunk_score:
                comment_df = get_top_chunk_df(
                    comment_df, video_id, comment_id, score_key=score_key)

            # Sort and get top 5
            top_df = comment_df.sort_values(by=score_key, ascending=False).head(TOP_KS[-1])

            target_df = comment_df[comment_df['target_idxs'] != "[]"]
            if target_df.empty:
                targets = []
            else:
                targets = json_load_str_list(target_df['target_idxs'].iloc[0])

            for top_k in TOP_KS:
                top_k_df = top_df.head(top_k)
                correct = False
                min_distance = None
                for source_idx, target_idxs in zip(top_k_df['source_idx'], top_k_df['target_idxs']):
                    # Min distance between predicted and target
                    if len(targets) == 0:
                        # Picked the wrong chunk, so no valid distance can be calculated
                        distance = None
                    else:
                        distance = min([abs(source_idx - target_idx) for target_idx in targets])
                        min_distance = distance if (min_distance is None or distance < min_distance) else min_distance
                    if source_idx in json_load_str_list(target_idxs):
                        correct = True
                        break

                result = {
                    'video_id': video_id,
                    'comment_id': comment_id,
                    'unique_id': f"{video_id}_{comment_id}",
                    'correct': correct,
                    'top_k': top_k,
                    'min_distance': min_distance
                }
                results.append(result)

    results_df = pd.DataFrame(results)

    for k in TOP_KS:
        top_k_df = results_df[results_df['top_k'] == k]
        print(f"Top {k} accuracy: {top_k_df['correct'].mean()}")
        print(f"Top {k} min distance: {top_k_df['min_distance'].mean()}")
    return results_df


if __name__ == '__main__':
    print(f"File: {fname}")
    for score_key in score_keys:
        print(f">> Score key: {score_key}")
        # print("No chunk score")
        run_score(df, with_chunk_score=False, score_key=score_key)
        # print("Chunk score")
        # run_score(df, with_chunk_score=True, score_key=score_key)
        print()
