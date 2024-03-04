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
import time
import tqdm
import sys
import editdistance
from rank_bm25 import BM25Okapi

sys.path.append(os.getcwd())
from scripts.evaluation import utils


# Seed
np.random.seed(0)
random.seed(0)

TOP_KS = [1, 3]


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--datasplit', type=str, default='test')
parser.add_argument('--score_method', type=str, default='')
parser.add_argument('--use_dataset_formatting', action='store_true', default=True)
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--skip_checking', action='store_true')
parser.add_argument('--print_examples', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--use_rolling_source_context', action='store_true')
parser.add_argument('--model', type=str, default='openaichat')
parser.add_argument('--max_tokens', type=int, default=500)
parser.add_argument('--window_size', type=int, default=20)
parser.add_argument('--position_t', type=str, default='end') # 'mid' 'start'
parser.add_argument('--num_removal_sentences', type=int, default=1) #  if use_removal is true, this is the number of sentences to remove
parser.add_argument('--top_k_responses', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()

# Use cuda if available
args.use_cuda = torch.cuda.is_available()
DEVICE = 'cuda' if args.use_cuda else 'cpu'

# CHATGPT_PROMPT_FNAME = f'prompts/{args.dataset}/chatgpt.txt'
# CHATGPT_PROMPT = open(CHATGPT_PROMPT_FNAME).read()

def model_init(model_string, cuda):
    if model_string.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_string)
        model = GPT2LMHeadModel.from_pretrained(model_string)
    elif model_string == 'gpt_j':
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    elif model_string == 'opt_6b':
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b",
                                             device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)
    else:
        raise ValueError()
    if cuda:
        model.to('cuda')
    model.eval()

    # If batch size > 1, we need to pad the input_ids
    if args.batch_size > 1:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

if not args.skip_checking:
    if args.method in ['gpt2', 'gpt_j', 'opt_6b']:
        model, tokenizer = model_init(args.method, args.use_cuda)

def get_per_token_logprobs(logprobs, input_ids):
    input_logprobs = []
    for i in range(1, input_ids.shape[1]):
        input_logprobs.append(logprobs[0, i-1, input_ids[0, i]].item())
    return input_logprobs

def sent_scoring_single_sentence(model_tokenizer, text, cuda):
    model = model_tokenizer[0]
    tokenizer = model_tokenizer[1]
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    if cuda:
        input_ids = input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    logits = outputs[1]
    logprobs = F.log_softmax(logits, dim=-1)
    input_logprobs = get_per_token_logprobs(logprobs, input_ids)
    logprob_query = sum(input_logprobs)
    result = {
        'logprob': logprob_query,
        'num_text_tokens': len(input_ids[0]),
    }
    return result

def sent_scoring(model_tokenizer, prefix, complete_text, cuda):
    model = model_tokenizer[0]
    tokenizer = model_tokenizer[1]
    prefix_input_ids = torch.tensor(tokenizer.encode(prefix, return_tensors='pt')).unsqueeze(0)
    input_ids = torch.tensor(tokenizer.encode(complete_text)).unsqueeze(0)  # Batch size 1
    prefix_length = prefix_input_ids.shape[-1]
    if cuda:
        prefix_input_ids = prefix_input_ids.to('cuda')
        input_ids = input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    # Use logits for scoring
    logits = outputs[1]
    logprobs = F.log_softmax(logits, dim=-1)
    input_logprobs = get_per_token_logprobs(logprobs, input_ids)
    prefix_length = prefix_input_ids.shape[-1] - 1 # -1 because we want prob of NEXT token
    logprob_query = sum(input_logprobs[prefix_length:])

    result = {
        'num_source_tokens': prefix_length,
        'raw_logprob': logprob_query,
        'logprob': logprob_query,
    }

    if args.normalize_probs:
        original_logprob_query = logprob_query
        logprob_query /= prefix_length
        print(f"Normalized logprob: {logprob_query} (original: {original_logprob_query})")
        result['logprob'] = logprob_query

    # args.normalize_probs
    return result

def get_causal_lm_score(query_text, source_text, separator='\n'):
    if not source_text: # just score query
        score = sent_scoring_single_sentence(
            (model, tokenizer),
            text=query_text,
            cuda=args.use_cuda
        )
    else:
        score = sent_scoring(
            (model, tokenizer),
            prefix=f"{source_text}{separator}",
            complete_text=f"{source_text}{separator}{query_text}",
            cuda=args.use_cuda
        )
    return score

def get_score_or_cache(probability_of, given, cache, separator):
    merged = given + separator + probability_of
    if merged in cache: 
        return cache[merged]
    else:
        score_dict = get_causal_lm_score(
            query_text=probability_of, 
            source_text=given, 
            separator=separator)
        score = score_dict['logprob'] 
        cache[merged] = score
        return score

def get_p_q_S(query_text, sources, prob_cache):
    # make sure the first sentence gets "Teacher: " prepended
    # copy sources
    S = sources.copy()
    S[0] = {"text": "Teacher: " + S[0]['text']}

    q_S = " ".join([s['text'] for s in S]) + "\n" + query_text

    # Check if we already have this in the cache
    if q_S in prob_cache:
        print(f"Found in cache")
        return prob_cache[q_S]
    
    log_p_q_S = 0
    # Otherwise calculate. p(q|s_N): 
    log_p_q_S_n = get_score_or_cache(
        probability_of=query_text, 
        given=sources[-1]['text'],
        cache=prob_cache, 
        separator='\n'
    )

    log_p_q_S += log_p_q_S_n
    for probability_i in range(len(sources) - 1, 0, -1):
        # p(S_i | S_{i-H:i-1})
        S_i = sources[probability_i]['text']
        S_imH_to_im1 = " ".join([sources[j]['text'] for j in range(max(0, probability_i - args.window_size), probability_i)])
        log_p_S_i_given_S_i_H_to_i_1 = get_score_or_cache(
            probability_of=S_i,
            given=S_imH_to_im1,
            cache=prob_cache,
            separator=' '
        )
        log_p_q_S += log_p_S_i_given_S_i_H_to_i_1

    log_p_S_0_dict = get_causal_lm_score(
        query_text=sources[0]['text'],
        source_text=""
    )
    log_p_S_0 = log_p_S_0_dict['logprob']
    log_p_q_S += log_p_S_0

    prob_cache[q_S] = log_p_q_S
    return log_p_q_S


def get_sliding_window_score(query_text, source_text, misc, query, source, source_idx, sources, prob_cache):
    # first calculate p(q|S) = p(q|S_n) p(S_n | S_{n-H:n-1}) ... p(S_2|S_1, S_0) p(S_1| S_0)
    p_q_S = get_p_q_S(query_text, sources, prob_cache)

    # # then calculate p(q|S \ s_t) = p(q|S_n, S_{t+1} | S_{t-H-1, ..., t-3, t-1}, ...)
    # remove s_t from S
    S_without_s_t = sources.copy()
    S_without_s_t.pop(source_idx)
    p_q_S_without_s_t = get_p_q_S(query_text, S_without_s_t, prob_cache)

    contribution = p_q_S - p_q_S_without_s_t
    # print(f"Contribution: {contribution} for {sources[source_idx]} to query {query_text}")

    return -(contribution)

def load_model():
    if args.method == 'edit_distance' or args.method == 'bm25':
        model=None
    elif args.method == 'semantic_similarity':
        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device=DEVICE)
    elif args.method == 'minilm':
        model = SentenceTransformer('all-MiniLM-L12-v2', device=DEVICE)
    elif args.method == 'cross_encoder':
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=DEVICE)
    return model

def get_bm25_score(query, sources, source_idx, prob_cache):
    """
    # Look up query_text in probe_cache -> if there, look up len(sources) -> if there, return scores[source_idx]
    # Otherwise, calculate BM25
    """
    query_text = query['text'].lower()
    if query_text in prob_cache:
        if len(sources) in prob_cache[query_text]:
            return prob_cache[query_text][len(sources)][source_idx]

    corpus = [source['text'].lower() for source in sources]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query_text.split(" ")
    scores = bm25.get_scores(tokenized_query)

    if query_text not in prob_cache:
        prob_cache[query_text] = dict()
    prob_cache[query_text][len(sources)] = scores
    return scores[source_idx]

def get_score(query_text, source_text, misc, query, source, source_idx, sources, prob_cache, model):
    """Return scores where higher is better"""
    # Use cuda if availabel
    device =  'cuda' if torch.cuda.is_available() else 'cpu'
    score_dict = dict()
    if args.method == 'edit_distance':
        distance = editdistance.eval(query_text, source_text)
        score_dict['prediction'] = -distance
    elif args.method == 'bm25':
        score = get_bm25_score(query, sources, source_idx, prob_cache)
        score_dict['prediction'] = score
    elif args.method == 'semantic_similarity':
        query_embedding = model.encode(query_text, convert_to_tensor=True).cuda()
        source_embedding = model.encode(source_text, convert_to_tensor=True).cuda()
        similarity = query_embedding.dot(source_embedding).item()
        score_dict['prediction'] = similarity
    elif args.method == 'minilm':
        query_embedding = model.encode(query_text, convert_to_tensor=True).cuda()
        source_embedding = model.encode(source_text, convert_to_tensor=True).cuda()
        similarity = query_embedding.dot(source_embedding).item()
        score_dict['prediction'] = similarity
    elif args.method == 'cross_encoder':
        cross_inp = [[query_text, source_text]]
        cross_scores = model.predict(cross_inp)
        score_dict['prediction'] = cross_scores[0]
    elif args.method in ['gpt2', 'gpt_neox', 'gpt_j', 'opt_6b']:
        if args.use_full_window_removal:
            # Sliding window
            print("Full sliding window")
            score = get_sliding_window_score(
                query_text=query_text, 
                source_text=source_text,
                misc=misc,
                query=query,
                source=source,
                source_idx=source_idx,
                sources=sources,
                prob_cache=prob_cache
            )
        else:
        # if args.use_window_removal:
            score_ = get_causal_lm_score(
                query_text=query_text, 
                source_text=source_text)
            score = -score_['logprob']

            # Track number of tokens
            for k, v in score_.items():
                score_dict[k] = v

        score_dict['prediction'] = score
        # # print(f"\log P(q|S \ s_t) = {score}")

        # p_q_s_t = get_causal_lm_score(query_text, misc['s_t'])
        # # print(f"\log P(q|s_t) = {p_q_s_t}")

        # p_s_t = get_causal_lm_score(misc['s_t'], "")
        # # print(f"\log P(s_t) = {p_s_t}")

        # p_q_S_with_s_t = get_causal_lm_score(query_text, misc['S_with_s_t'])
        # # print(f"\log P(q|S_with_s_t) = {p_q_S_with_s_t}")

        # p_q_S_without_s_t = get_causal_lm_score(query_text, misc['S_without_s_t'])
        # # print(f"\log P(q|S_without_s_t) = {p_q_S_without_s_t}")

        # p_S_with_s_t = get_causal_lm_score(misc['S_with_s_t'], "")
        # # print(f"\log P(S_with_s_t) = {p_S_with_s_t}")

        # p_S_without_s_t = get_causal_lm_score(misc['S_without_s_t'], "")
        # # print(f"\log P(S_without_s_t) = {p_S_without_s_t}")
        # score_dict['\log p(q|s_t)'] = p_q_s_t
        # score_dict['\log p(s_t)'] = p_s_t
        # score_dict['\log p(q|S_with_s_t)'] = p_q_S_with_s_t
        # score_dict['\log p(q|S_without_s_t)'] = p_q_S_without_s_t # This should be the same as prediction
        # score_dict['\log p(S_with_s_t)'] = p_S_with_s_t
        # score_dict['\log p(S_without_s_t)'] = p_S_without_s_t

    return score_dict

def run_random_baseline(df, query_dir, source_dir, output_fname):
    """
    Run random baseline on dataset
    """
    for query_fname in os.listdir(query_dir):
        with open(os.path.join(query_dir, query_fname)) as f:
            queries = json.load(f)

            if isinstance(queries, dict):
                queries = [queries]

        with open(os.path.join(source_dir, query_fname)) as f:
            sources = json.load(f)

        for query_idx, query in enumerate(queries):
            source_idx = 0

            if check_if_result_already_exists(df, query_fname, query_idx, source_idx, query, sources[source_idx]):
                continue

            # Randomly pick a source sentence idx
            idx = np.random.randint(len(sources))
            target_idxs = utils.get_target_idxs(query=query, dataset=args.dataset)

            # Pick 3 random sentences
            idxs = np.random.choice(len(sources), 3, replace=False)

            if None in target_idxs:
                continue

            result = {
                'fname': query_fname,
                'query': query,
                'query_idx': query_idx,
                'source': sources[idx],
                'source_idx': source_idx,
                'target_idxs': target_idxs,
                'prediction_top1': idx,
                'prediction_top3': idxs.tolist(),
                'predicted_text': sources[idx],
            }
            # Append to df
            df = pd.concat([df, pd.DataFrame([result])])
            query_idx += 1
    df.to_csv(output_fname, index=False)
    return df


def calculate_random_baseline_metrics(df):
    # If target_idxs is a string, convert to list
    if isinstance(df.iloc[0].target_idxs, str):
        df['target_idxs'] = df['target_idxs'].apply(lambda x: json.loads(x))
    if isinstance(df.iloc[0].prediction_top3, str):
        df['prediction_top3'] = df['prediction_top3'].apply(lambda x: json.loads(x))

    # Calculate correct: prediction in target_idxs
    df['correct'] = df.apply(lambda row: row['prediction_top1'] in row['target_idxs'], axis=1)
    df['min_distance'] = df.apply(lambda row: min([abs(target_idx - row['prediction_top1']) for target_idx in row['target_idxs']]), axis=1)

    df['correct_top3'] = df.apply(lambda row: any([idx in row['target_idxs'] for idx in row['prediction_top3']]), axis=1)
    df['min_distance_top3'] = df.apply(lambda row: min([
        min([abs(target_idx - idx) for target_idx in row['target_idxs']]) for idx in row['prediction_top3']]), axis=1)
    # Round to 2 decimal places
    acc = round(df['correct'].mean(), 2)
    print(f"Random baseline accuracy: {acc}")
    min_distance = round(df['min_distance'].mean(), 2)
    print(f"Random baseline min distance: {min_distance}")

    acc_top3 = round(df['correct_top3'].mean(), 2)
    print(f"Random baseline accuracy top 3: {acc_top3}")
    min_distance_top3 = round(df['min_distance_top3'].mean(), 2)
    print(f"Random baseline min distance top 3: {min_distance_top3}")

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

def format_query_source_text(query, source, source_idx, sources, use_rolling_source_context=None, use_removal=False):
    misc = dict()
    if use_rolling_source_context is None:
        use_rolling_source_context = args.use_rolling_source_context

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
            source_text = format_removal_space_sep_sight(sources, source_idx)
            source_without = source_text['without']
            source_with = source_text['with']
            source_text = source_without
            # Track s_t, s_{t-k:t} or whatever window, s_{t-k:t} \ s_t 
            misc['source_without'] = source_without
            misc['source_with'] = source_with
            misc['s_t'] = format_dataset_text(sources[source_idx])
        else:
            source_text = format_dataset_text(source)
    query_text = format_dataset_text(query)
    return query_text, source_text, misc


def run_baseline(df, query_dir, source_dir, output_fname):
    model = load_model()
    for query_fname in tqdm.tqdm(os.listdir(query_dir)):
        prob_cache = dict()
        with open(os.path.join(query_dir, query_fname)) as f:
            queries = json.load(f)

            if isinstance(queries, dict):
                queries = [queries]

        with open(os.path.join(source_dir, query_fname)) as f:
            sources = json.load(f)

        for query_idx, query in tqdm.tqdm(enumerate(queries)):
            target_idxs = utils.get_target_idxs(query=query, dataset=args.dataset)

            if None in target_idxs:
                continue

            for source_idx, source in enumerate(sources):
                query_text, source_text, misc = format_query_source_text(
                    query=query,
                    source=source,
                    source_idx=source_idx,
                    sources=sources)

                # Unique id: fname, query_idx, source_idx
                if not args.overwrite and check_if_result_already_exists(
                    df, query_fname,
                    query_idx, source_idx,
                    {"text":  query_text},
                    {"text": source_text}):
                    continue

                # Calculate edit distance between query and source
                prediction = get_score( 
                    query_text=query_text,
                    source_text=source_text,
                    misc=misc,
                    query=query,
                    source=source,
                    source_idx=source_idx,
                    sources=sources, 
                    prob_cache=prob_cache,
                    model=model
                )

                result = {
                    'fname': query_fname,
                    'query': query,
                    'query_idx': query_idx,
                    'source': source,
                    'source_idx': source_idx,
                    'query_text': query_text,
                    # 'source_text': source_text, # take out source text to save space
                    'target_idxs': target_idxs,
                    # 'normalized_prediction': normalized,
                    **prediction
                }
                # Append to df
                df = pd.concat([df, pd.DataFrame([result])])
                # Save df
                df.to_csv(output_fname, index=False)
                # results.append(result)
            query_idx += 1
    return df


def calculate_baseline_metrics(df):
    # For each query, take the source with the highest edit distance. Check if it's in target_idxs
    results_df = []
    for fname in df['fname'].unique():
        fname_df = df[df['fname'] == fname]
        for query_text in fname_df['query_text'].unique():
        # for query_idx in df['query_idx'].unique():
            query_df = fname_df[fname_df['query_text'] == query_text]
            # Check that there's only 1 fname
            try:
                assert len(query_df['fname'].unique()) == 1
            except:
                import pdb; pdb.set_trace()

            query_df = query_df.sort_values(by='prediction', ascending=False)
            # print(f"Edit distance: {query_df.iloc[0].prediction}")
            # Get the source_idx top_k
            for top_k in TOP_KS:
                source_indices = query_df.iloc[:top_k].source_idx.values
                # Get the target_idxs
                target_idxs = query_df.iloc[0].target_idxs
                if isinstance(target_idxs, str):
                    target_idxs = json.loads(target_idxs)

                correct = any([source_idx in target_idxs for source_idx in source_indices])

                # Take the distance of the top_k to the target_idxs
                distance = min([abs(target_idx - source_idx) for source_idx in source_indices for target_idx in target_idxs])

                results_df.append({
                    'fname': fname,
                    'query_text': query_text,
                    'correct': correct, 
                    'top_k': top_k, 
                    'distance': distance
                })

    results_df = pd.DataFrame(results_df)

    for top_k in TOP_KS:
        acc = round(results_df[results_df['top_k'] == top_k]['correct'].mean(), 2)
        print(f"Accuracy top {top_k}: {acc}")
        dist = round(results_df[results_df['top_k'] == top_k]['distance'].mean(), 2)
        print(f"Min distance top {top_k}: {dist}")
    return results_df

def print_random_examples(df):
    # Print examples
    for query_idx in df['query_idx'].unique():
        query_df = df[df['query_idx'] == query_idx]
        query = query_df.iloc[0].query['text']
        predicted_source_idx = query_df.iloc[0].prediction
        source = query_df[query_df['source_idx'] == predicted_source_idx].iloc[0].source['text']

        print(f"Predicted source: {source}")
        print(f"Query: {query}")
        print()

def print_examples(df):
    # Print examples
    num_printed = 0
    for fname in df['fname'].unique():
        fname_df = df[df['fname'] == fname]
        for query_text in fname_df['query_text'].unique():
        # for query_idx in df['query_idx'].unique():
            query_df = fname_df[fname_df['query_text'] == query_text]
            query = query_df.iloc[0].query_text

            query_df = query_df.sort_values(by='prediction', ascending=False)
            # print(f"Edit distance: {query_df.iloc[0].prediction}")
            # Get the source_idx top_k
            top_responses = query_df.iloc[:5]
            # source_idx = query_df.iloc[0].source_idx
            # print(f"Source idx: {source_idx} in target_idxs: {query_df.iloc[0].target_idxs}")
            # Get the target_idxs
            target_idxs = query_df.iloc[0].target_idxs
            if isinstance(target_idxs, str):
                target_idxs = json.loads(target_idxs)

            correct = any([source_idx in target_idxs for source_idx in top_responses.source_idx.values])

            target_sentences = []
            for target_idx in target_idxs:
                source_text = query_df[query_df['source_idx'] == target_idx].iloc[0].source
                # source_text = json.loads(source_text)['text']
                target_sentences.append(source_text)
            
            target_sentence_scores = []
            for target_idx in target_idxs:
                prediction = query_df[query_df['source_idx'] == target_idx].iloc[0].prediction
                target_sentence_scores.append(prediction)

            predicted_sentences = []
            for source in top_responses.source.values:
                # source_text = json.loads(source)
                predicted_sentences.append(source)

            predicted_scores = top_responses.prediction.values

            print("----------------")
            print(f"Query: {query}")
            print(f"Correct: {correct}")
            max_score = query_df.prediction.max()
            min_score = query_df.prediction.min()
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

            print("* All sentences:")
            sorted_query_df = query_df.sort_values(by='source_idx', ascending=True)
            # Print sentence scores 
            for sentence, score in zip(sorted_query_df.source.values, sorted_query_df.prediction.values):
                print(f">>>> [{score}] {sentence}")            

            print()
             
            num_printed += 1

            if num_printed > 10:
                break


def prompt_model(prompt, manifest):
    success = False
    while not success:
        try:
            # Call LM and retry if invalid response
            lm_response = manifest.run(
                prompt,
                temperature=0.0,
                max_tokens=args.max_tokens
            )
            success = True
        except:
            print('Invalid response from LM. Retrying...')
            # Add a delay to avoid rate limiting
            time.sleep(10)
    return lm_response

def check_if_result_already_exists(df, fname, query_idx, source_idx, query, source):
    """
    Check if result already exists in df
    """
    if len(df) == 0:
        return False
    if args.dataset == 'dailydialog' or args.dataset == 'reccon': # there's only one query per file
        return len(df[(df['fname'] == fname) & (df['source_idx'] == source_idx)]) > 0
    # if the method isn't random, then check query_text, source_text
    if args.method != 'random':
        return len(df[(df['fname'] == fname) & (df['query_text'] == query['text']) & (df['source_idx'] == source_idx)]) > 0
    elif args.method == 'chatgpt':
        return len(df[(df['fname'] == fname) & (df['query_text'] == query['text'])]) > 0
    # return len(df[(df['fname'] == fname) & (df['query'] == query) & (df['source'] == source)]) > 0
    return len(df[(df['fname'] == fname) & (df['query_idx'] == query_idx) & (df['source_idx'] == source_idx)]) > 0


if __name__ == '__main__':
    query_dirs, source_dirs, output_fname = utils.get_dirs_and_output_fname(args)
    print(f"Output fname: {output_fname}")

    # Load df if it exists
    if os.path.exists(output_fname):
        df = pd.read_csv(output_fname, engine='python', encoding='utf-8', on_bad_lines='skip')
    else:
        df = pd.DataFrame()

    for (query_dir, source_dir) in zip(query_dirs, source_dirs):
        if args.method == 'random':
            if not args.skip_checking:
                print(query_dir, source_dir)
                df = run_random_baseline(
                    df=df,
                    query_dir=query_dir,
                    source_dir=source_dir,
                    output_fname=output_fname
                    )
            calculate_random_baseline_metrics(df)
        else: # sentence by sentence
            if not args.skip_checking:
                if args.batch_size == 1:
                    df = run_baseline(
                        df=df,
                        query_dir=query_dir,
                        source_dir=source_dir,
                        output_fname=output_fname
                    )
            # If it's empty, don't calculate metrics, print that it's empty
            if len(df) > 0:
                calculate_baseline_metrics(df)
            else:
                print("Empty df!")

            if args.print_examples:
                print_examples(df)
