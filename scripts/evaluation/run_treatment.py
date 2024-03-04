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
import torch
import torch.nn.functional as F
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForCausalLM
)
import numpy as np
import tqdm
import sys
import csv

sys.path.append(os.getcwd())
from scripts.evaluation import utils

# Seed
np.random.seed(0)
random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--score_method', type=str, default='')
parser.add_argument('--collect_logprobs', action='store_true')
parser.add_argument('--analyze_logprobs', action='store_true')
parser.add_argument('--print_examples', action='store_true')

parser.add_argument('--datasplit', type=str, default='test')
parser.add_argument('--use_dataset_formatting', action='store_true', default=True)
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--use_prob_cache', action='store_true')
parser.add_argument('--skip_checking', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--model', type=str, default='openaichat')
parser.add_argument('--max_tokens', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()

# Use cuda if available
args.use_cuda = torch.cuda.is_available()
print(f"Using CUDA? {args.use_cuda}")

def model_init(model_string, cuda):
    if model_string.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_string)
        model = GPT2LMHeadModel.from_pretrained(model_string)
    elif model_string == 'gpt_j':
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            # torch_dtype=torch.float32,
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True
            )
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    elif model_string == 'opt_6b':
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-6.7b",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-6.7b",
            use_fast=False
        )
    else:
        raise ValueError()
    if cuda:
        print('SETTING TO CUDA')
        model.to('cuda')
    model.eval()

    # If batch size > 1, we need to pad the input_ids
    if args.batch_size > 1:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_per_token_logprobs(logprobs, input_ids):
    input_logprobs = []
    for i, input_id in enumerate(input_ids[0]):
        logprob = logprobs[0, i, input_id].item()
        # input_logprobs.append(logprobs[0, i-1, input_ids[0, i]].item())
        input_logprobs.append(logprob)
    return input_logprobs

def score_text(model_tokenizer, score_dict, prob_cache):
    model = model_tokenizer[0]
    tokenizer = model_tokenizer[1]

    score_entire_sequence = (score_dict['prefix_text'] == score_dict['suffix_text'])
    
    # Tokenize
    complete_text = score_dict['complete_text']

    if not complete_text: # empty string
        score_dict['loss'] = None
        score_dict['logprob'] = None
        score_dict['normalized_logprob'] = None
        return score_dict

    cache_key = score_dict['shorthand'] + " " + complete_text

    if args.use_prob_cache and cache_key in prob_cache:
        score_dict['loss'] = prob_cache[cache_key]['loss']
        score_dict['logprob'] = prob_cache[cache_key]['logprob']
        score_dict['normalized_logprob'] = prob_cache[cache_key]['normalized_logprob']
        score_dict['prefix_length'] = prob_cache[cache_key]['prefix_length']
        score_dict['suffix_length'] = prob_cache[cache_key]['suffix_length']
        score_dict['complete_text_length'] = prob_cache[cache_key]['complete_text_length']
        return score_dict   

    # Prefix includes the separator
    prefix_text = score_dict['prefix_text'] + score_dict['separator']
    prefix_input_ids = torch.tensor(tokenizer.encode(prefix_text, return_tensors='pt')).unsqueeze(0)
    suffix_input_ids = torch.tensor(tokenizer.encode(score_dict['suffix_text'], return_tensors='pt')).unsqueeze(0)
    complete_input_ids = torch.tensor(tokenizer.encode(complete_text)).unsqueeze(0)  # Batch size 1
    
    # Get lengths
    prefix_length = prefix_input_ids.shape[-1]
    suffix_length = suffix_input_ids.shape[-1]
    complete_text_length = complete_input_ids.shape[-1]

    # Track lengths
    score_dict['prefix_length'] = prefix_length
    score_dict['suffix_length'] = suffix_length
    score_dict['complete_text_length'] = complete_text_length


    if args.use_cuda:
        prefix_input_ids = prefix_input_ids.to('cuda')
        complete_input_ids = complete_input_ids.to('cuda')

    # Check if context fits
    max_tokens = 1024 if args.method == 'gpt2' else 2048
    if complete_text_length > max_tokens:
        score_dict['context_fits'] = False
        score_dict['loss'] = None
        score_dict['logprob'] = None
        score_dict['normalized_logprob'] = None
        return score_dict



    with torch.no_grad():
        outputs = model(complete_input_ids, labels=complete_input_ids)
        score_dict['context_fits'] = True

    # Alternatively use logits
    logits = outputs[1]
    logprobs = F.log_softmax(logits, dim=-1)
    input_logprobs = get_per_token_logprobs(logprobs, complete_input_ids[:, 1:]) # exclude first token because model doesn't predict it
    if score_entire_sequence:
        logprob = sum(input_logprobs)
        normalized_length = len(input_logprobs)
        normalized_logprob = logprob / normalized_length
    else:
        logprob = sum(input_logprobs[prefix_length-1:]) # -1 because input_logprobs excludes first token
        normalized_length = len(input_logprobs[prefix_length-1:])
        normalized_logprob = logprob / normalized_length

    score_dict['loss'] = outputs.loss.item()
    score_dict['logprob'] = logprob
    score_dict['normalized_logprob'] = normalized_logprob

    # Cache
    prob_cache[cache_key] = {
        'loss': outputs.loss.item(),
        'logprob': logprob,
        'normalized_logprob': normalized_logprob,
        'prefix_length': prefix_length,
        'suffix_length': suffix_length,
        'complete_text_length': complete_text_length,
    }

    # if score_dict['shorthand']=='p(q|S \ s_t)':
    #     import pdb;  pdb.set_trace()

    return score_dict

def format_query(text_dict, include_dataset_formatting=True):
    if include_dataset_formatting:
        if args.dataset == 'dailydialog':
            text = f"Speaker {text_dict['speaker']}: {text_dict['text']}"
        elif args.dataset == 'sight':
            text = "Student: " + text_dict['text']
        elif args.dataset in ['squad', 'inquisitive']:
            text = f"Question: {text_dict['text']}"
        elif args.dataset == 'reccon':
            text = f"Speaker {text_dict['speaker']} (emotion = {text_dict['emotion']}): {text_dict['text']}"
    else:
        text = text_dict['text']
    return text

def format_source(text_dict, include_dataset_formatting=True):
    if include_dataset_formatting:
        if args.dataset == 'dailydialog' or args.dataset == 'reccon':
            text = f"Speaker {text_dict['speaker']}: {text_dict['text']}"
        elif args.dataset == 'sight':
            text = f"""A teacher is teaching a class, and a student asks a question.

            Teacher:  {text_dict['text']}"""
        elif args.dataset == 'squad':
            text = f"Text: {text_dict['text']}"
        elif args.dataset == 'inquisitive':
            text = f"Text: {text_dict['text']}"
    else:
        text = text_dict['text']
    return text


def format_sources(sources, include_dataset_formatting=True):
    # list of dictionaries
    separator = ' ' if (args.dataset not in ['dailydialog', 'reccon']) else '\n'
    source_text = ''
    for idx, source in enumerate(sources):
        if idx == 0:
            if include_dataset_formatting:
                source_text += format_source(source)
            else:
                source_text += format_source(source, include_dataset_formatting=False)
        else:
            if args.dataset == 'dailydialog' or args.dataset == 'reccon':
                source_text += separator + format_source(source)
            else:
                source_text += separator + format_source(source, include_dataset_formatting=False)
    return source_text

def get_texts_to_score(sources, source_idx, query, score_methods):
    """
    return list of dictionaries: 
    [
        {
            "prefix": prefix_text, # eg source (should be formatted already)
            "suffix": suffix_text, # eg query
            "seperator": separator, # eg \n
            "prefix_length": prefix_length, # num tokens, do this when scoring
            "suffix_length": suffix_length, # num tokens, do this when scoring
            "complete_text": complete_text, # eg source + separator + query
            "complete_text_length": complete_text_length, # num tokens, do this when scoring
            "prefix_shorthand": prefix_shorthand, # str of what prefix represents
            "suffix_shorthand": suffix_shorthand, # str of what suffix represents
            "shorthand": shorthand, # str of what the whole thing represents eg p(q|s_t)
        }
    ]
    TODO add more context in SIGHT
    """
    texts = []
    shorthand2text = dict()

    to_score = [ # format: (prefix, suffix)
        # # priors
        # ('q', 'q'), # p(q)
        # ('s_t', 's_t'), # p(s_t)
        # ('S', 'S'), # p(S)
        # ('S \ s_t', 'S \ s_t'), # p(S \ s_t)
        # ('S_{<=t}', 'S_{<=t}'), # p(S_{<=t})
        # ('S_{<=t} \ s_t', 'S_{<=t} \ s_t'), # p(S_{<=t} \ s_t)
        # # joint
        # ('S q', 'S q'), # p(S, q)
        # ('s_t q', 's_t q'), # p(s_t, q)
        # ('S \ s_t q', 'S \ s_t q'), # p(S \ s_t, q)
        # ('S_{<=t} q', 'S_{<=t} q'), # p(S_{<=t}, q)
        # ('S_{<=t} \ s_t q', 'S_{<=t} \ s_t q'), # p(S_{<=t} \ s_t, q)
        # # conditionals
        # ('s_t', 'q'), # p(q|s_t)
        # ('S', 'q'), # p(q|S)
        # ('S \ s_t', 'q'), # p(q|S \ s_t)
        # ('S_{<=t}', 'q'), # p(q|S_{<=t})
        # ('S_{<=t} \ s_t', 'q'), # p(q|S_{<=t} \ s_t)
    ]

    # # Prior
    # shorthand2text['q'] = format_query(query)
    # shorthand2text['S'] = format_sources(sources)
    # shorthand2text['s_t'] = format_source(sources[source_idx])
    # shorthand2text['S \ s_t'] = format_sources(sources[:source_idx] + sources[source_idx+1:])
    # shorthand2text['S_{<=t}'] = format_sources(sources[:source_idx+1])
    # shorthand2text['S_{<=t} \ s_t'] = format_sources(sources[:source_idx])
    # # Joint
    # shorthand2text['S q'] = shorthand2text['S'] + '\n' + shorthand2text['q']
    # shorthand2text['s_t q'] = shorthand2text['s_t'] + '\n' + shorthand2text['q']
    # shorthand2text['S \ s_t q'] = shorthand2text['S \ s_t'] + '\n' + shorthand2text['q']
    # shorthand2text['S_{<=t} q'] = shorthand2text['S_{<=t}'] + '\n' + shorthand2text['q']
    # shorthand2text['S_{<=t} \ s_t q'] = shorthand2text['S_{<=t} \ s_t'] + '\n' + shorthand2text['q']

    if 'single-sentence' in score_methods:
        to_score += [('s_t', 'q')] # p(q|s_t)
        shorthand2text['q'] = format_query(query)
        shorthand2text['s_t'] = format_source(sources[source_idx])
    
    if 'auto-regressive' in score_methods:
        to_score += [('S_{<=t}', 'q')] # p(q|S_{<=t})
        shorthand2text['q'] = format_query(query)
        shorthand2text['S_{<=t}'] = format_sources(sources[:source_idx+1])

    if 'causal-full' in score_methods:
        to_score += [('S', 'q')]
        to_score += [('S \ s_t', 'q')]
        shorthand2text['q'] = format_query(query)
        shorthand2text['S'] = format_sources(sources)
        shorthand2text['S \ s_t'] = format_sources(sources[:source_idx] + sources[source_idx+1:])

    if 'causal-end' in score_methods:
        to_score += [('S_{<=t}', 'q')]
        to_score += [('S_{<=t} \ s_t', 'q')]
        shorthand2text['q'] = format_query(query)
        shorthand2text['S_{<=t}'] = format_sources(sources[:source_idx+1])
        shorthand2text['S_{<=t} \ s_t'] = format_sources(sources[:source_idx])


    for prefix_shorthand, suffix_shorthand in to_score:
        result = {
            "prefix_text": shorthand2text[prefix_shorthand],
            "suffix_text": shorthand2text[suffix_shorthand],
            "t": source_idx,
            "T": len(sources),
            "separator": '\n', # use only when prefix != suffix
            "prefix_length": None, # do when scoring
            "suffix_length": None,
            "complete_text_length": None,
            "prefix_shorthand": prefix_shorthand,
            "suffix_shorthand": suffix_shorthand,
            "shorthand": f"p({suffix_shorthand}|{prefix_shorthand})" if prefix_shorthand != suffix_shorthand else f"p({suffix_shorthand})",
        }
        if prefix_shorthand == suffix_shorthand:
            result['complete_text'] = shorthand2text[prefix_shorthand]
        else:
            result['complete_text'] = shorthand2text[prefix_shorthand] + '\n' + shorthand2text[suffix_shorthand]
        texts.append(result)
    return texts


def check_and_filter_results(df, texts, fname):
    """
    Can check fname, prefix short hand, suffix shorthand, shorthand, complete_text
    """
    # if df is empty, reutnr all
    if len(df) == 0:
        return texts
    new_texts = []
    for text in texts:
        prefix_shorthand = text['prefix_shorthand']
        suffix_shorthand = text['suffix_shorthand']
        shorthand = text['shorthand']
        complete_text = text['complete_text']

        # Check if fname exists
        if len(df[df['fname'] == fname]) > 0:
            fname_df = df[df['fname'] == fname]
            # Check if prefix, suffix, shorthand, complete_text exists
            if len(fname_df[
                (fname_df['prefix_shorthand'] == prefix_shorthand) & 
                (fname_df['suffix_shorthand'] == suffix_shorthand) & 
                (fname_df['shorthand'] == shorthand) & 
                (fname_df['complete_text'] == complete_text)]) > 0:
                continue

        new_texts.append(text)
    return new_texts

def get_scores(score_dicts, prob_cache, model, tokenizer):
    for score_dict in score_dicts:
        # Modifies score_dict
        score_text(
            model_tokenizer=(model, tokenizer),
            score_dict=score_dict,
            prob_cache=prob_cache
        )
    return score_dicts

def already_exists(df, query_fname, query_idx, source_idx):
    if all([c_name in df.columns for c_name in ['fname', 'query_idx', 'source_idx']]):
        return len(df[(df['fname'] == query_fname) & (df['query_idx'] == query_idx) & (df['source_idx'] == source_idx)]) > 0
    return False

def run_baseline(df, query_dir, source_dir, output_fname, score_methods, model, tokenizer):
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
                # print(f"{source_idx}/{len(sources)}")

                # Check if already exists by fname, query_idx, source_idx
                if not args.overwrite:
                    if already_exists(
                            df=df,
                            query_fname=query_fname,
                            query_idx=query_idx,
                            source_idx=source_idx):
                        continue

                texts = get_texts_to_score(
                    sources=sources,
                    source_idx=source_idx,
                    query=query,
                    score_methods=score_methods
                )
                if not texts:
                    continue

                texts_with_scores = get_scores(
                    score_dicts=texts,
                    prob_cache=prob_cache,
                    model=model,
                    tokenizer=tokenizer
                )

                for text_with_score in texts_with_scores:
                    result = {
                        'fname': query_fname,
                        'query': query,
                        'query_idx': query_idx,
                        'source': source,
                        'source_idx': source_idx,
                        'target_idxs': target_idxs,
                        'is_target': source_idx in target_idxs,
                        **text_with_score
                    }
                    df = pd.concat([df, pd.DataFrame([result])])
                # save with utf-8 encoding
                df.to_csv(output_fname, index=False, encoding='utf-8')
    return df

def calculate_baseline_metrics(df, score_methods):
    """
    
    """
    # methods = [
    #     'single-sentence', 'auto-regressive', 'causal-full', 'causal-end',
    #     # 'causal-full-2nd',
    #     # 'raw-causal-full-2nd'
    #     ]
    top_ks = [1,3]
    scores_df = utils.get_scores_df(df, score_methods)
    results_df = utils.get_results_df(scores_df, methods=score_methods, top_ks=top_ks)
    utils.print_topk_scores(results_df, methods=score_methods, top_ks=top_ks)

    if args.print_examples:
        for method in score_methods:
            utils.print_examples(scores_df, prediction_column=method, max_examples=10)


def load_df(fname):
    if os.path.exists(fname):
        df = pd.read_csv(
            fname, 
            engine='python', 
            encoding='utf-8',
            encoding_errors='ignore',
            on_bad_lines='skip')
    else:
        df = pd.DataFrame()
    return df


if __name__ == '__main__':
    if not args.skip_checking and args.method in ['gpt2', 't5', 'ul2', 'gpt_j', 'gpt_neox', 'opt_6b']:
        model, tokenizer = model_init(args.method, args.use_cuda)
    else:
        model, tokenizer = None, None

    query_dirs, source_dirs, output_fname = utils.get_dirs_and_output_fname(args)
    print(f"Output fname: {output_fname}")

    if not args.score_method:
        score_methods = ['single-sentence', 'auto-regressive', 'causal-full']
    else:
        score_methods = [args.score_method]

    if args.collect_logprobs:
        # If the output name already exists, load
        if os.path.exists(output_fname):
            # df = pd.read_csv(output_fname, engine='python', encoding='utf-8', on_bad_lines='skip')
            df = pd.read_csv(output_fname, on_bad_lines='skip') #quoting=csv.QUOTE_NONE)
            print(f"File exists, shape: {df.shape}")
        else:
            df = pd.DataFrame()
        for (query_dir, source_dir) in zip(query_dirs, source_dirs):
            df = run_baseline(
                df=df,
                query_dir=query_dir,
                source_dir=source_dir,
                output_fname=output_fname,
                score_methods=score_methods,
                model=model,
                tokenizer=tokenizer
            )
    
    if args.analyze_logprobs:
        df = load_df(output_fname)
        calculate_baseline_metrics(df, score_methods)
        # print_examples(df)
