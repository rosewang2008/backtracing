import pandas as pd
import json
import os

def get_target_idxs(query, dataset):
    if dataset == 'dailydialog' or dataset == 'reccon':
        return query['source_sentences']
    elif dataset == 'squad':
        if query['line number'] is None:
            return [None]
        return [query['line number']-1] # 1-indexed
    elif dataset == 'sight':
        return query['annotatedSourceSentencesIndices']
    elif dataset == 'inquisitive':
        return [query['sentence_id']-1]
        

def get_dirs_and_output_fname(args):
    query_dirs = []
    source_dirs = []
    if args.dataset in ['dailydialog', 'sight', 'inquisitive']:
        query_dir = f'data/{args.dataset}/query/{args.datasplit}'
        source_dir = f'data/{args.dataset}/sources/{args.datasplit}'
        query_dirs.append(query_dir)
        source_dirs.append(source_dir)
    elif args.dataset == 'reccon':
        query_dir = f'data/{args.dataset}/query/'
        source_dir = f'data/{args.dataset}/sources/'
        query_dirs.append(query_dir)
        source_dirs.append(source_dir)
    elif args.dataset == 'squad':
        # Read all the dirs in the query dir
        for query_dir in os.listdir(f'data/{args.dataset}/query/'):
            query_dirs.append(f'data/{args.dataset}/query/{query_dir}')
            source_dirs.append(f'data/{args.dataset}/sources/{query_dir}')

    print(f"Query dirs: {query_dirs}")
    print(f"Source dirs: {source_dirs}")
    if args.method == 'rerank':
        method = 'minilm'
        output_fname = f'results/{args.dataset}/{args.datasplit}/{method}'
    elif not args.score_method:
        output_fname = f'results/{args.dataset}/{args.datasplit}/{args.method}'
    else:
        output_fname = f'results/{args.dataset}/{args.datasplit}/{args.method}_{args.score_method}'

    output_fname += '.csv'

    # Make directories if they don't exist
    os.makedirs(os.path.dirname(output_fname), exist_ok=True)

    args.exp_name = output_fname.split("/")[-1]
    print(f"Output fname: {output_fname}")

    return query_dirs, source_dirs, output_fname


def print_examples(df, prediction_column='prediction', max_examples=10):
    # Print examples
    num_examples = 0
    for fname in df['fname'].unique():
        fname_df = df[df['fname'] == fname]
        for query_text in fname_df['query'].unique():
        # for query_idx in df['query_idx'].unique():
            query_df = fname_df[fname_df['query'] == query_text]
            query = query_df.iloc[0].query

            query_df = query_df.sort_values(prediction_column, ascending=False)

            # Get the source_idx top_k
            top_responses = query_df.iloc[:3]

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
                prediction = query_df[query_df['source_idx'] == target_idx].iloc[0][prediction_column]
                target_sentence_scores.append(prediction)

            predicted_sentences = []
            for source in top_responses.source.values:
                # source_text = json.loads(source)
                predicted_sentences.append(source)

            predicted_scores = top_responses[prediction_column].values

            print("----------------")
            print(f"Query: {query}")
            print(f"Correct: {correct}")
            max_score = query_df[prediction_column].max()
            min_score = query_df[prediction_column].min()
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
            for sentence, score in zip(sorted_query_df.source.values, sorted_query_df[prediction_column].values):
                print(f">>>> [{score}] {sentence}")            

            print()

            num_examples += 1
            if num_examples >= max_examples:
                break



def print_topk_scores(df, methods, top_ks):
    """
    Usage:

    >> df = pd.read_csv(DATA_FNAME)
    >> methods = ['single-sentence', 'auto-regressive', 'causal-full', 'causal-end']
    >> top_ks = [1,3,5]
    >> scores_df = utils.get_scores_df(df)
    >> results_df = utils.get_results_df(scores_df, methods=methods, top_ks=top_ks)
    >> utils.print_topk_scores(results_df, methods=methods, top_ks=top_ks)

    """
    for method in methods:
        for top_k in top_ks:
            results_df_subset = df[(df['method'] == method) & (df['top_k'] == top_k)]
            accuracy = results_df_subset.correct.mean()
            # Round accuracy to 2 decimal places
            accuracy = round(accuracy, 2)
            distance = results_df_subset.min_distance.mean()
            distance = round(distance, 2)
            print(f"Method: {method}, top_k: {top_k}, accuracy: {accuracy}, distance: {distance}")
        
        print()
    pass

def get_scores_df(df, methods):
    results_df = pd.DataFrame()
    num_errors = 0
    num_large_context = 0
    for fname in df['fname'].unique():
        fname_df = df[df['fname'] == fname]
        for query_text in fname_df['query'].unique():
            query_df = fname_df[fname_df['query'] == query_text]
            # Remove nan
            query_df = query_df[~query_df['source_idx'].isna()]
            for source_idx in query_df['source_idx'].unique():
                source_df = query_df[query_df['source_idx'] == source_idx]

                # Check if df contains context_fits, and if context_fits = False, skip
                if 'context_fits' in source_df.columns:
                    if not source_df.iloc[0].context_fits:
                        num_large_context += 1
                        continue
                # score_dict['context_fits']

                source_text = source_df.iloc[0].source

                if not (len(source_df['query'].unique()) == 1 and len(source_df['source'].unique()) == 1):
                    import pdb; pdb.set_trace()

                result = {
                    'fname': fname,
                    'query': query_text,
                    'source': source_text,
                    'source_idx': source_idx,
                    'target_idxs': source_df.iloc[0].target_idxs,
                    'is_target': source_df.iloc[0].is_target   
                }
                
                if 'single-sentence' in methods:
                    p_q_given_s_t = source_df[source_df['shorthand'] == 'p(q|s_t)']
                    if len(p_q_given_s_t) == 0:
                        num_errors += 1
                        continue
                    p_q_given_s_t = p_q_given_s_t.iloc[0].normalized_logprob
                    result['single-sentence'] = p_q_given_s_t

                if 'auto-regressive' in methods:
                    p_q_given_S_ltt = source_df[source_df['shorthand'] == 'p(q|S_{<=t})']
                    if len(p_q_given_S_ltt) == 0:
                        num_errors += 1
                        continue
                    p_q_given_S_ltt = p_q_given_S_ltt.iloc[0].normalized_logprob
                    result['auto-regressive'] = p_q_given_S_ltt

                if 'causal-full' in methods:
                    p_q_given_S = source_df[source_df['shorthand'] == 'p(q|S)']
                    p_q_given_S_st = source_df[source_df['shorthand'] == 'p(q|S \ s_t)']

                    if len(p_q_given_S) == 0 or len(p_q_given_S_st) == 0:
                        num_errors += 1
                        continue
                    
                    p_q_given_S = p_q_given_S.iloc[0].normalized_logprob
                    p_q_given_S_st = p_q_given_S_st.iloc[0].normalized_logprob
                    result['causal-full'] = p_q_given_S - p_q_given_S_st
                
                results_df = pd.concat([results_df, pd.DataFrame([result])])
    results_df = pd.DataFrame(results_df)
    print(f"Total: {len(df)}")
    print(f"Num errors: {num_errors}")
    print(f"Num large context: {num_large_context}")
    return results_df

def get_results_df(df, methods, top_ks):
    """
    For each method calculate its accuracy
    """
    # For each query, get the source with the highest score & report accuracy for each method
    results_df = pd.DataFrame()
    for fname in df['fname'].unique():
        fname_df = df[df['fname'] == fname]
        for query_text in fname_df['query'].unique():
            query_df = fname_df[fname_df['query'] == query_text]

            target_idxs = query_df.iloc[0].target_idxs
            if isinstance(target_idxs, str):
                target_idxs = json.loads(target_idxs)


            for method in methods:
                method_df = query_df.sort_values(by=method, ascending=False)
                for top_k in top_ks:
                    top_k_df = method_df.iloc[:top_k]

                    # correct = any(top_k_df.is_target.values)
                    correct = any([source_idx in target_idxs for source_idx in top_k_df.source_idx.values])

                    # Min distance
                    min_distance = min([abs(source_idx - target_idx) for source_idx in top_k_df.source_idx.values for target_idx in target_idxs])

                    result = {
                        'fname': fname,
                        'query': query_text,
                        'method': method,
                        'top_k': top_k,
                        'correct': correct, 
                        'min_distance': min_distance
                    }

                    # Report N, P, predicted P, predicted N
                    # Actual negative
                    N = len(top_k_df[top_k_df['is_target'] == False])
                    # Actual positive
                    P = len(top_k_df[top_k_df['is_target'] == True])
                    # Predicted negative

                    results_df = pd.concat([results_df, pd.DataFrame([result])]) 
    return results_df

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k"):
    """Returns the number of tokens used by a list of messages."""
    import tiktoken
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-16k":
        tokens_per_message = None # TODO
        tokens_per_name = -1
    elif model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens