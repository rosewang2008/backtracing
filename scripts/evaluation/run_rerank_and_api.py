"""
Runs evaluation of different methods on datasets

Store results as results/<dataset>/<method>.json
"""

import argparse
import json
import os
import random
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import torch.nn.functional as F
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForCausalLM
)
import numpy as np
import openai
import tqdm
import sys
import editdistance
import ast
import time

sys.path.append(os.getcwd())
from scripts.evaluation import utils

# Seed
np.random.seed(0)
random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--datasplit', type=str, default='test')
parser.add_argument('--score_method', type=str, default='')
parser.add_argument('--use_dataset_formatting', action='store_true', default=True)
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--skip_checking', action='store_true')
parser.add_argument('--print_examples', action='store_true')
parser.add_argument('--analyze', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--use_rolling_source_context', action='store_true')
parser.add_argument('--normalize_probs', action='store_true')
parser.add_argument('--use_full_window_removal', action='store_true') # this is for shapley
parser.add_argument('--use_window_removal', action='store_true') # this is for shapley
parser.add_argument('--use_argmin_removal', action='store_true') # this is for shapley
parser.add_argument('--dailydialog_evaluation_metric', type=str, default='turn_tm1') # turn_tm1 (t minus 1)
parser.add_argument('--model', type=str, default='gpt-3.5-turbo-16k') # previously default='openaichat'
parser.add_argument('--max_tokens', type=int, default=500)
parser.add_argument('--window_size', type=int, default=20)
parser.add_argument('--position_t', type=str, default='end') # 'mid' 'start'
parser.add_argument('--num_removal_sentences', type=int, default=1) #  if use_removal is true, this is the number of sentences to remove
parser.add_argument('--top_k_responses', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()
# Use cuda if available
args.use_cuda = torch.cuda.is_available()

CHATGPT_PROMPT_FNAME = f'prompts/{args.dataset}/chatgpt.txt'
CHATGPT_PROMPT = open(CHATGPT_PROMPT_FNAME).read()
TOP_KS = [1, 3]

def format_dataset_text(text_dict):
    use_dataset_formatting = args.use_dataset_formatting
    if use_dataset_formatting:
        if args.dataset == 'dailydialog':
            text = f"Speaker {text_dict['speaker']}: {text_dict['text']}"
        elif args.dataset == 'reccon':
            if 'source_sentences' in text_dict.keys():
                text = f"Speaker {text_dict['speaker']} (emotion = {text_dict['emotion']}): {text_dict['text']}"
            else:
                text = f"Speaker {text_dict['speaker']}: {text_dict['text']}"
        elif args.dataset == 'peerread':
            is_review_comment = not('section_sentence_num' in text_dict)
            if is_review_comment:
                if use_dataset_formatting:
                    text = "Reviewer: " + text_dict['text']
                else:
                    text = text_dict['text']
            else: # is paper text
                if text_dict['section_sentence_num'] == 0:
                    text = f"Section {text_dict['section']}: {text_dict['text']}"
                else:
                    text = text_dict['text']
        elif args.dataset == 'inquisitive':
            is_query = 'sentence_id' in text_dict
            if is_query:
                text = f"Question: {text_dict['text']}"
            else:
                text = f"Text: {text_dict['text']}"
        elif args.dataset == 'sight':
            is_student_comment = "annotatedSourceSentencesIndices" in text_dict
            if is_student_comment:
                if use_dataset_formatting:
                    text = "Student: " + text_dict['text']
                else:
                    text = text_dict['text']
            else: # is lecture text
                text = "Teacher: " + text_dict['text']
        elif args.dataset == 'squad':
            is_query = 'line number' in text_dict
            text = text_dict['text']
            if is_query:
                text = f"Question: {text}"
            else:
                text = f"Text: {text}"
        else:
            raise NotImplementedError
    else:
        text = text_dict['text']
    return text


def format_rolling_context_space_sep(sources, start_idx, end_idx):
    source_text = format_dataset_text(sources[start_idx])
    # Just add all the other sentences, with spaces
    if end_idx > start_idx:
        for i in range(start_idx+1, end_idx+1):
            source_text += ' ' + sources[i]['text']
    return source_text

def format_removal_space_sep_sight(sources, source_idx):
    """
    Let's say the window_size is 5 and the source_idx is 3: 

                             |  source_idx    
                             v   
    x_{t-2}     x_{t-1}     x_t     x_{t+1}     x_{t+2}  

    The arguments we care about are 
    - args.window_size (window size), 
    - args.position_t (whether we put source_idx at the beginning, middle, or end of the window)
    - args.num_removal_sentences (how many sentences we remove from the window, including the source_idx!)
    """

    num_sentences_to_remove = args.num_removal_sentences - 1 # includes source_idx already
    remove_idx = num_sentences_to_remove // 2

    assert num_sentences_to_remove >= 0

    # 1. First determine range of indices to include and exclude
    if args.position_t == 'begin':
        start_idx = source_idx
        end_idx = source_idx + args.window_size
    elif args.position_t == 'mid':
        added_idx = args.window_size // 2
        start_idx = source_idx - added_idx
        end_idx = source_idx + added_idx
    elif args.position_t == 'end':
        start_idx = source_idx - args.window_size
        end_idx = source_idx

    start_exclude_idx = source_idx - remove_idx 
    end_exclude_idx = source_idx + remove_idx

    # Make sure start_idx and end_idx are valid
    start_idx = max(0, start_idx)
    end_idx = min(len(sources) - 1, end_idx)
    start_exclude_idx = max(0, start_exclude_idx)
    end_exclude_idx = min(len(sources) - 1, end_exclude_idx)

    # Now create the ranges 
    include_range = list(range(start_idx, end_idx+1))
    exclude_range = list(range(start_exclude_idx, end_exclude_idx+1))
    # print(f"source_idx: {source_idx}, include_range: {include_range}, exclude_range: {exclude_range}")

    assert source_idx in exclude_range

    source_text_without = ''
    source_text_with = ''
    for i in include_range:
        # we want to exclude the source_idx
        if i not in exclude_range:
            if source_text_without != '':
                source_text_without += ' ' + sources[i]['text']
            else: # first sentence
                source_text_without += format_dataset_text(sources[i])

        # we want to include the source_idx
        if source_text_with != '':
            source_text_with += ' ' + sources[i]['text']
        else: # first sentence
            source_text_with += format_dataset_text(sources[i])
    return {"with" : source_text_with, "without" : source_text_without}

def format_removal_space_sep(sources, start_idx, end_idx, source_idx):
    with_source_text = ''
    without_source_text = ''
    for i in range(start_idx, end_idx+1):
        if i != source_idx:
            if without_source_text != '':
                without_source_text += ' ' + sources[i]['text']
            else: # first sentence
                without_source_text += format_dataset_text(sources[i])

        if with_source_text != '':
            with_source_text += ' ' + sources[i]['text']
        else: # first sentence
            with_source_text += format_dataset_text(sources[i])
    return {"with" : with_source_text, "without" : without_source_text}

def format_query_source_text(query, source, source_idx, sources, use_rolling_source_context=None, use_removal=None):
    misc = dict()
    if use_rolling_source_context is None:
        use_rolling_source_context = args.use_rolling_source_context
    if use_removal is None:
        use_removal = args.use_full_window_removal or args.use_argmin_removal or args.use_window_removal

    if args.dataset == 'dailydialog' or args.dataset == 'reccon':
        if use_rolling_source_context:
            source_text = '' # Accumulate text up to and including source_idx
            for i in range(source_idx + 1):
                source_text += format_dataset_text(sources[i])
                # Add newline if not the last source
                if i != source_idx:
                    source_text += '\n'
        elif use_removal:
            # concatenate the source text EXCEPT for the source_idx
            source_text = ''
            for i in range(len(sources)):
                if i != source_idx:
                    source_text += format_dataset_text(sources[i])
                    # Add newline if not the last source
                    if i != len(sources) - 1:
                        source_text += '\n'
        else:
            source_text = format_dataset_text(source)
    elif args.dataset in ['squad', 'inquisitive'] or (args.dataset == 'sight' and 'oracle' in args.datasplit):
        if use_rolling_source_context:
            source_text = format_rolling_context_space_sep(sources, 0, source_idx)
        elif use_removal:
            source_text_dict = format_removal_space_sep(sources, 0, len(sources)-1, source_idx)
            source_text = source_text_dict['without']
            # without_source_text = source_text_dict['without']
            # Track s_t, s_{t-k:t} or whatever window, s_{t-k:t} \ s_t 
            misc['s_t'] = format_dataset_text(sources[source_idx])
            misc['S_with_s_t'] = source_text_dict['with']
            misc['S_without_s_t'] = source_text_dict['without']
        else:
            source_text = format_dataset_text(source)
    elif args.dataset == 'sight':
        if use_rolling_source_context:
            source_text = format_rolling_context_space_sep(
                sources, max(0, source_idx - args.window_size), source_idx)
        elif use_removal:
            raise ValueError("Should not be used ")
        else:
            source_text = format_dataset_text(source)
    query_text = format_dataset_text(query)
    return query_text, source_text, misc

def rerank(query_df, df):
    device =  'cuda' if torch.cuda.is_available() else 'cpu'
    query_df = query_df.sort_values(by='prediction', ascending=False)[:5]
    for idx, row in query_df.iterrows():
        query_text = row['query_text']
        if args.dataset == 'inquisitive':
            source_json = ast.literal_eval(row['source'])
            source_text = 'Text: ' + source_json['text']
        elif args.dataset == 'dailydialog' or args.dataset == 'reccon':
            source_json = ast.literal_eval(row['source'])
            source_text = f"Speaker {source_json['speaker']}: {source_json['text']}"
        elif args.dataset == 'sight':
            source_json = ast.literal_eval(row['source'])
            source_text = 'Teacher: ' + source_json['text']
        if not query_df.loc[query_df['source_idx'] == row['source_idx'], 'prediction_rerank'].isna().values:
            continue

        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        cross_inp = [[query_text, source_text]]
        cross_scores = cross_encoder.predict(cross_inp)
        rerank_score = cross_scores[0]
        query_df.loc[query_df['source_idx'] == row['source_idx'], 'prediction_rerank'] = rerank_score
        df.loc[(df['source_idx'] == row['source_idx']) & (df['query_text'] == query_text), 'prediction_rerank'] = rerank_score
    return query_df


def calculate_baseline_metrics(df):
    if args.method == 'rerank':
        rerank_path = f'results/{args.dataset}/{args.datasplit}/{args.method}.csv'
        if os.path.exists(rerank_path):
            df = pd.read_csv(rerank_path)
        else:
            df['prediction_rerank'] = None
        
    # For each query, take the source with the highest edit distance. Check if it's in target_idxs
    results_df = []
    for fname in tqdm.tqdm(df['fname'].unique()):
        fname_df = df[df['fname'] == fname]
        for query_text in tqdm.tqdm(fname_df['query_text'].unique()):
        # for query_idx in df['query_idx'].unique():
            query_df = fname_df[fname_df['query_text'] == query_text]
            # Check that there's only 1 fname
            try:
                assert len(query_df['fname'].unique()) == 1
            except:
                import pdb; pdb.set_trace()

            if args.use_window_removal or args.use_full_window_removal or args.use_argmin_removal: # Take the source with the lowest score (contribution of removing that source)
                query_df = query_df.sort_values(by='prediction', ascending=True)
            elif args.method == 'rerank':
                query_df = rerank(query_df, df)
                query_df =  query_df.sort_values(by='prediction_rerank', ascending=False)
            else: # Take the source with the highest score
                query_df = query_df.sort_values(by='prediction', ascending=False)
            # Get the source_idx top_k
            for top_k in TOP_KS:
                source_indices = query_df.iloc[:top_k].source_idx.values
                # Get the target_idxs
                target_idxs = query_df.iloc[0].target_idxs
                if isinstance(target_idxs, str):
                    target_idxs = json.loads(target_idxs)

                correct = any([source_idx in target_idxs for source_idx in source_indices])

                # Distance of the predicted_idxs to the target_idxs
                distance = min([abs(source_idx - target_idx) for source_idx in source_indices for target_idx in target_idxs])

                results_df.append({
                    'fname': fname,
                    'query_text': query_text,
                    'correct': correct, 
                    'top_k': top_k,
                    'distance': distance
                })
            if args.method == 'rerank':
                df.to_csv(rerank_path, index=False)
    results_df = pd.DataFrame(results_df)

    for top_k in TOP_KS:
        raw_acc = results_df[results_df['top_k'] == top_k]['correct'].mean()
        acc = round(raw_acc, 3)
        print(f"Accuracy top {top_k}: {acc}")
        dist = results_df[results_df['top_k'] == top_k]['distance'].mean()
        dist = round(dist, 3)
        print(f"Distance: {dist }")
    return results_df


def print_examples(df):
    PREDICTION_COLUMN = 'prediction_rerank' if args.method == 'rerank' else 'prediction'
    num_examples = 0
    # Print examples
    for fname in df['fname'].unique():
        fname_df = df[df['fname'] == fname]
        for query_text in fname_df['query_text'].unique():
            query_df = fname_df[fname_df['query_text'] == query_text]

            # Set all nan predictions to be -np.inf; not in top-k ranks
            query_df[PREDICTION_COLUMN] = query_df[PREDICTION_COLUMN].fillna(-np.inf)

            query = query_df.iloc[0].query_text

            if args.use_full_window_removal or args.use_window_removal: # Take the source with the lowest score (contribution of removing that source)
                query_df = query_df.sort_values(by=PREDICTION_COLUMN, ascending=True)
            else: # Take the source with the highest score
                # query_df = query_df.sort_values(by=PREDICTION_COLUMN, ascending=False)
                query_df =  query_df.sort_values(by=PREDICTION_COLUMN, ascending=False)
            # Get the source_idx top_k
            top_responses = query_df.iloc[:3]
            # Get the target_idxs
            target_idxs = query_df.iloc[0].target_idxs
            if isinstance(target_idxs, str):
                target_idxs = json.loads(target_idxs)

            correct = any([source_idx in target_idxs for source_idx in top_responses.source_idx.values])

            target_sentences = []
            for target_idx in target_idxs:
                try:
                    # source_text = query_df[query_df['source_idx'] == target_idx].iloc[0].source
                    source_text = fname_df[fname_df['source_idx'] == target_idx].iloc[0].source
                except:
                    import pdb; pdb.set_trace()
                # source_text = json.loads(source_text)['text']
                target_sentences.append(source_text)
            
            target_sentence_scores = []
            for target_idx in target_idxs:
                # prediction = query_df[query_df['source_idx'] == target_idx].iloc[0][PREDICTION_COLUMN]
                prediction = fname_df[fname_df['source_idx'] == target_idx].iloc[0][PREDICTION_COLUMN]
                target_sentence_scores.append(prediction)

            predicted_sentences = []
            for source in top_responses.source.values:
                # source_text = json.loads(source)
                predicted_sentences.append(source)

            predicted_scores = top_responses[PREDICTION_COLUMN].values

            print("----------------")
            print(f"Query: {query}")
            print(f"Correct: {correct}")
            max_score = query_df[PREDICTION_COLUMN].max()
            min_score = query_df[PREDICTION_COLUMN].min()
            print(f"Max score: {max_score}")
            print(f"Min score: {min_score}")

            # Print top response
            print("* Target sentences:")
            for target_sentence, target_score in zip(target_sentences, target_sentence_scores):
                print(f">>>> [{target_score}] {target_sentence}")
            
            print()
            print("* Predicted sentences:")
            for predicted_sentence, predicted_score in zip(predicted_sentences, predicted_scores):
                print(f">>>> [{predicted_score}] {predicted_sentence}")

            print()

            # print("* All sentences:")
            # sorted_query_df = query_df.sort_values(by='source_idx', ascending=True)
            # # Print sentence scores 
            # for sentence, score in zip(sorted_query_df.source.values, sorted_query_df[PREDICTION_COLUMN].values):
            #     print(f">>>> [{score}] {sentence}")            
            # print()

            num_examples += 1

            if num_examples == 10:
                break


def setup_api_key():
    openai.api_key = os.environ["OPENAI_API_KEY"]

def check_if_result_already_exists(df, fname, query_idx, source_idx, query, source):
    """
    Check if result already exists in df
    """
    if len(df) == 0:
        return False
    if args.dataset == 'dailydialog' or args.dataset == 'reccon': # there's only one query per file
        return len(df[(df['fname'] == fname) & (df['source_idx'] == source_idx)]) > 0
    # if the method isn't random, then check query_text, source_text
    if args.method == 'chatgpt':
        return len(df[(df['fname'] == fname) & (df['query_idx'] == query_idx)]) > 0 # & (df['query_text'] == query_text)
    if args.method != 'random':
        return len(df[(df['fname'] == fname) & (df['query_text'] == query['text']) & (df['source_idx'] == source_idx)]) > 0
    # return len(df[(df['fname'] == fname) & (df['query'] == query) & (df['source'] == source)]) > 0
    return len(df[(df['fname'] == fname) & (df['query_idx'] == query_idx) & (df['source_idx'] == source_idx)]) > 0

def get_model_response(model, messages, temperature):
    need_to_retry = True
    while need_to_retry:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                )
            print("Response:")
            print(response) # NOTE You can check whether the predicted # is aligned with the line numbers in the text
            response_content = response["choices"][0]["message"]["content"]
            prompt_tokens = response["usage"]["prompt_tokens"]
            print(response_content)
            print(f"Prompt tokens: {prompt_tokens}") 
            need_to_retry = False
        # InvalidRequestError
        except openai.error.InvalidRequestError:
            print("InvalidRequestError")
            response_content = ""
            prompt_tokens = -1
            need_to_retry = False
        except openai.error.RateLimitError:
            # Wait 10 seconds
            print("RateLimitError")
            time.sleep(10)
    return response_content

def run_chatgpt_baseline(df, query_dir, source_dir, output_fname):
    """
    Run chatgpt baseline on dataset
    """
    setup_api_key()
    
    for query_fname in tqdm.tqdm(os.listdir(query_dir)):
        print(f"Query fname {query_fname}")
        with open(os.path.join(query_dir, query_fname)) as f:
            queries = json.load(f)

        with open(os.path.join(source_dir, query_fname)) as f:
            sources = json.load(f)

        # if args.dataset == 'sight':
        #     queries = [queries]
        if not isinstance(queries, list):
            queries = [queries]

        print(f"Number of queries: {len(queries)}")

        query_sub_idx = 0
        for query in queries:
            print(f"Query sub idx: {query_sub_idx}/{len(queries)}")
            target_idxs = utils.get_target_idxs(query=query, dataset=args.dataset)
            transcript = ''

            for source_idx, source in enumerate(sources):
                transcript += f"{source_idx}. {format_dataset_text(source)}\n"

            # REmove last newline
            transcript = transcript[:-1]
            query_text = f"{format_dataset_text(query)}"

            if args.dataset == 'dailydialog':
                query_idx = int(query_fname.split('.')[0]) # works only for daily dialog since daily dialog has 1 query per fname
            elif args.dataset == 'reccon':
                query_idx = query_fname.split('.')[0] # works only for daily dialog since daily dialog has 1 query per fname
            else:
                query_idx = query_fname.split('.')[0]
                query_idx = f'{query_idx}-{query_sub_idx}'

            if check_if_result_already_exists(df, query_fname, query_idx, source_idx, query, source):
                print(f"Result already exists for query {query_idx} and source {source_idx}")
                query_sub_idx += 1
                continue

            if args.dataset == 'reccon':
                prompt = CHATGPT_PROMPT.format(
                    transcript=transcript,
                    query=query_text, 
                    emotion=query['emotion']
                )
            else:
                prompt = CHATGPT_PROMPT.format(
                    transcript=transcript,
                    query=query_text
                )

            messages = [{"role": "user", "content": prompt}]

            print(">>>>>")
            # print(prompt)
            # prompt_tokens = utils.num_tokens_from_messages(messages, "gpt-3.5-turbo")
            # print(f"Prompt tokens: {prompt_tokens}") 
            # import pdb; pdb.set_trace()

            response_content = get_model_response(model=args.model, messages=messages, temperature=0)
    
            result = {
                'fname': query_fname,
                'prompt': prompt,
                'query': query,
                'query_idx': query_idx,
                # 'query_text': query_text, 
                'source': source,
                'source_idx': source_idx,
                'target_idxs': target_idxs,
                'prediction': response_content
            }

            # Append to df
            df = pd.concat([df, pd.DataFrame([result])])
            df.to_csv(output_fname, index=False)

            query_sub_idx += 1
    return df

def calculate_chatgpt_baseline_metrics(df):
    # For each query, take the source with the highest edit distance. Check if it's in target_idxs
    COLUMN_KEY = 'fname' if args.dataset in ['dailydialog', 'reccon'] else 'query_idx'
    results_df = []
    num_max_context_errors = 0
    for query_idx in df[COLUMN_KEY].unique():
        query_df = df[df[COLUMN_KEY] == query_idx]
        # Check there's only one prediction
        try:
            assert len(query_df) == 1
        except:
            import pdb; pdb.set_trace()
        # Parse prediction jsonlines. If prediction is NaN then error
        prediction = query_df.iloc[0].prediction
        if pd.isna(prediction):
            num_max_context_errors += 1
            continue
        else:
            prediction = json.loads(query_df.iloc[0].prediction)

        predicted_idxs = [result['line number'] for result in prediction]
        # Get the target_idxs
        target_idxs = json.loads(query_df.iloc[0].target_idxs) # convert '[1, 2, 3]' to [1, 2, 3]

        # Correct if any of the predicted_idxs are in target_idxs
        correct = len(set(predicted_idxs).intersection(set(target_idxs))) > 0

        # Take distance of the predicted_idxs to the target_idxs
        distance = min([abs(predicted_idx - target_idx) for predicted_idx in predicted_idxs for target_idx in target_idxs])

        results_df.append({
            'query_idx': query_idx,
            'correct': correct,
            'distance': distance
        })

    results_df = pd.DataFrame(results_df)
    # Round to 2 decimal places
    acc = round(results_df['correct'].mean(), 2)
    print(f"Accuracy: {acc}")
    # Print distance
    print(f"Distance: {results_df['distance'].mean()}")
    print(f"Number of max context errors: {num_max_context_errors}")



if __name__ == '__main__':
    query_dirs, source_dirs, output_fname = utils.get_dirs_and_output_fname(args)
    print(f"Output fname: {output_fname}")

    # Load df if it exists
    if os.path.exists(output_fname):
        df = pd.read_csv(output_fname, engine='python', encoding='utf-8', on_bad_lines='skip')
    else:
        df = pd.DataFrame()

    for (query_dir, source_dir) in zip(query_dirs, source_dirs):
        if args.method == 'chatgpt':
            if not args.skip_checking:
                df = run_chatgpt_baseline(
                    df=df,
                    query_dir=query_dir,
                    source_dir=source_dir,
                    output_fname=output_fname
                    )
            calculate_chatgpt_baseline_metrics(df)
        else: # sentence by sentence
            # If it's empty, don't calculate metrics, print that it's empty
            if len(df) > 0:
                if args.print_examples:
                    # replace fname 'minilm' with 'rerank' and check if it exists
                    rerank_fname = output_fname.replace('minilm', 'rerank')
                    if os.path.exists(rerank_fname):
                        df = pd.read_csv(rerank_fname)
                        print_examples(df)
                    
                if args.analyze:
                    calculate_baseline_metrics(df)
            else:
                print("Empty df!")
