"""
Analysis script on 
- Location of the target indices
- Similarity between the target and query

"""
import argparse
import json
import os
import random
import numpy as np
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import seaborn as sns
import sys
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util

sys.path.append(os.getcwd())
from scripts.evaluation import utils

# Seed
np.random.seed(0)
random.seed(0)


# Pretty plots
# sns.set_style("darkgrid")
sns.set_context("paper")
# Use textsc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="")
parser.add_argument('--method', type=str, default='edit_distance')
parser.add_argument('--score_method', type=str, default='')
parser.add_argument('--datasplit', type=str, default='test') # Or annotated for SIGHT
parser.add_argument('--plot_locations', action='store_true', default=False)
parser.add_argument('--plot_similarity', action='store_true', default=False)
parser.add_argument('--report_dataset_statistics', action='store_true', default=False)
args = parser.parse_args()


dataset2plotname = {
    'reccon': 'Conversation',
    'sight': 'Lecture',
    'inquisitive': 'News Article',
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def collect_locations(query_dirs, source_dirs, args):
    # get 'location', 'max_location'
    # location: the index of the target in the source
    results = []
    num_queries = 0
    for (query_dir, source_dir) in zip(query_dirs, source_dirs):
        for query_fname in tqdm.tqdm(os.listdir(query_dir)):
            with open(os.path.join(query_dir, query_fname)) as f:
                queries = json.load(f)

            with open(os.path.join(source_dir, query_fname)) as f:
                sources = json.load(f)

            if not isinstance(queries, list):
                queries = [queries]

            num_queries += len(queries)

            for query in queries:
                target_idxs = utils.get_target_idxs(query=query, dataset=args.dataset)

                for target_idx in target_idxs:
                    results.append({
                        'location': target_idx, 
                        'max_location': len(sources) - 1, # because it's 0-indexed
                        'perc_location': target_idx / (len(sources) - 1),
                        'dataset': args.dataset,
                    })
    print(f"{args.dataset}: Number of queries: {num_queries}")
    return results


def plot_locations(args):  
    datasets = ['sight', 'inquisitive', 'reccon']
    num_datasets = len(datasets)
    results = []
    for dataset in datasets:
        args.dataset = dataset
        query_dirs, source_dirs, _ = utils.get_dirs_and_output_fname(args)
        results += collect_locations(query_dirs, source_dirs, args)
    results = pd.DataFrame(results)

    # Plot: have three plots, one for each dataset.
    # Plot histogram in buckets of 0.1 by perc_location
    fig, ax = plt.subplots(num_datasets, 1, figsize=(6, 6), sharex=True)
    for i, dataset in enumerate(datasets):
        sns.histplot(
            data=results[results['dataset'] == dataset], 
            x='perc_location', 
            ax=ax[i], 
            binrange=(0, 1),
            bins=10,
        )
        ax[i].set_title(dataset2plotname[dataset])
    plt.tight_layout()
    # Rename x-axis to Location of Cause
    plt.xlabel('Location of Cause')
    output_path = "plots/target_location.pdf"
    plt.savefig(output_path)


def collect_similarity(query_dirs, source_dirs, args):
    """
    Track 
    - the similarity between the query and the target,
    - the max similarity between the query and the source
    """
    model = SentenceTransformer('all-MiniLM-L12-v2', device=DEVICE)
    results = []
    num_queries = 0
    for (query_dir, source_dir) in zip(query_dirs, source_dirs):
        for query_fname in tqdm.tqdm(os.listdir(query_dir)):
            print(f"Query fname {query_fname}")
            with open(os.path.join(query_dir, query_fname)) as f:
                queries = json.load(f)

            with open(os.path.join(source_dir, query_fname)) as f:
                sources = json.load(f)

            source_texts = [_['text'] for _ in sources]

            if not isinstance(queries, list):
                queries = [queries]

            print(f"Number of queries: {len(queries)}")

            for query in queries:
                target_idxs = utils.get_target_idxs(query=query, dataset=args.dataset)
                query_text = [query['text']]
                query_embedding = model.encode(query_text, convert_to_tensor=True).cuda()
                source_embedding = model.encode(source_texts, convert_to_tensor=True).cuda()
                similarity = torch.matmul(query_embedding, source_embedding.T).cpu().numpy()[0] # num_queries x num_sources
                # Get max similarity with source
                max_similarity = np.max(similarity)

                for target_idx in target_idxs:
                    gt_similarity = similarity[target_idx]
                    kvs = [
                        ('gt_similarity', gt_similarity, 'GT Sim'),
                        ('max_similarity', max_similarity, 'Max Sim'),
                        ('similarity_diff', max_similarity - gt_similarity, 'Diff Max-GT'),
                    ]
                    for k, v, plot_name in kvs:
                        results.append({
                            'key': k,
                            'value': v,
                            'dataset': args.dataset,
                            'plot_name': plot_name,
                        })
    print(f"{args.dataset}: Number of queries: {num_queries}")
    return results


def plot_similarity(args):
    datasets = ['sight', 'inquisitive', 'reccon']
    num_datasets = len(datasets)
    results = []
    output_fname = "results/sight/similarity.csv"
 
    # Check if results already exist
    if os.path.exists(output_fname):
        results = pd.read_csv(output_fname)
    else:
        for dataset in datasets:
            args.dataset = dataset
            if dataset == 'sight':
                args.datasplit = 'annotated'
            else:
                args.datasplit = 'test'
            query_dirs, source_dirs, _ = utils.get_dirs_and_output_fname(args)
            results += collect_similarity(query_dirs, source_dirs, args)
        results = pd.DataFrame(results)
        # Save df to csv
        results.to_csv(output_fname)

    # Set three plots, one for each dataset.
    fig, ax = plt.subplots(1, num_datasets, figsize=(6, 6), sharey=True)
    type2plotname = {
        'gt_similarity': 'GT Sim',
        'max_similarity': 'Max Sim',
        'similarity_diff': 'Diff Max-GT',
    }
    for i, dataset in enumerate(datasets):
        sns.barplot(
            data=results[results['dataset'] == dataset], 
            x='plot_name', 
            y='value', 
            ax=ax[i],
            # Map type to plotname
            # hue='key',
        )
        ax[i].set_title(dataset2plotname[dataset])
        # Blank out x axis title
        ax[i].set_xlabel('')
        # Only label the y-axis for the first plot
        if i == 0:
            ax[i].set_ylabel('Similarity')
        else:
            ax[i].set_ylabel('')
    plt.tight_layout()
    # Rename x-axis to Location of Cause
    # plt.xlabel('Location of Cause')
    output_path = "plots/similarity.pdf"
    plt.savefig(output_path)

    # # Barplot of gt_similarity, max_similarity, and difference split by dataset
    # for similarity_type in ['gt_similarity', 'max_similarity', 'similarity_diff']:
    #     sns.barplot(
    #         data=results,
    #         x='dataset',
    #         y=similarity_type,
    #     )
    #     plt.ylabel(f'Similarity {similarity_type}')
    #     plt.xlabel('Dataset')
    #     plt.ylim(0, 1)
    #     plt.tight_layout()
    #     output_path = f"plots/{similarity_type}.pdf"
    #     plt.savefig(output_path)
    #     plt.clf()

def collect_dataset_statistics(query_dirs, source_dirs, args):
    # get 'location', 'max_location'
    # location: the index of the target in the source
    results = []
    num_queries = 0
    for (query_dir, source_dir) in zip(query_dirs, source_dirs):
        for query_fname in tqdm.tqdm(os.listdir(query_dir)):
            with open(os.path.join(query_dir, query_fname)) as f:
                queries = json.load(f)

            with open(os.path.join(source_dir, query_fname)) as f:
                sources = json.load(f)

            if not isinstance(queries, list):
                queries = [queries]

            num_queries += len(queries)

            for query in queries:
                results.append({
                    'type': 'query',
                    'num_sentences': len(query['text'].split('.')),
                    'num_words': len(query['text'].split()),
                    'dataset': args.dataset,
                })
            for source in sources:
                results.append({
                    'type': 'source',
                    'query_fname': query_fname,
                    'num_words': len(source['text'].split()),
                    'dataset': args.dataset,
                })
    return results

def report_dataset_statistics(args):
    """
    For queries, report:
    - Total number of queries
    - Mean number of words
    - Median number of words
    - Min number of words
    - Max number of words

    For sources, report:
    - Mean number of source sentences per fname
    - Median number of sentences
    - Min number of sentences
    - Max number of sentences
    - Mean number of words
    """

    datasets = ['sight', 'inquisitive', 'reccon']
    num_datasets = len(datasets)
    results = []
    for dataset in datasets:
        args.dataset = dataset
        if dataset == 'sight':
            args.datasplit = 'annotated'
        else:
            args.datasplit = 'test'
        query_dirs, source_dirs, _ = utils.get_dirs_and_output_fname(args)
        results += collect_dataset_statistics(query_dirs, source_dirs, args)
    results = pd.DataFrame(results)

    # Print statistics
    for dataset in datasets: 
        dataset_df = results[results['dataset'] == dataset]
        # Queries
        num_queries = len(dataset_df[dataset_df['type'] == 'query'])
        print(f"{dataset}: Number of queries: {num_queries}")
        words = dataset_df[dataset_df['type'] == 'query']['num_words']
        mean_num_words = words.mean()
        print(f"{dataset}: Mean number of words: {mean_num_words}")
        median_num_words = words.median()
        print(f"{dataset}: Median number of words: {median_num_words}")
        min_num_words = words.min()
        print(f"{dataset}: Min number of words: {min_num_words}")
        max_num_words = words.max()
        print(f"{dataset}: Max number of words: {max_num_words}")

        # Sources
        source_df = dataset_df[dataset_df['type'] == 'source']
        num_sources = len(source_df)
        print(f"{dataset}: Number of sources: {num_sources}")
        num_sources_per_fname = source_df.groupby('query_fname').size()
        mean_num_sources_per_fname = num_sources_per_fname.mean()
        print(f"{dataset}: Mean number of sources per fname: {mean_num_sources_per_fname}")
        median_num_sources_per_fname = num_sources_per_fname.median()
        print(f"{dataset}: Median number of sources per fname: {median_num_sources_per_fname}")
        min_num_sources_per_fname = num_sources_per_fname.min()
        print(f"{dataset}: Min number of sources per fname: {min_num_sources_per_fname}")
        max_num_sources_per_fname = num_sources_per_fname.max()
        print(f"{dataset}: Max number of sources per fname: {max_num_sources_per_fname}")
        words = source_df['num_words']
        mean_num_words = words.mean()
        print(f"{dataset}: Mean number of words: {mean_num_words}")
        median_num_words = words.median()
        print(f"{dataset}: Median number of words: {median_num_words}")
        min_num_words = words.min()
        print(f"{dataset}: Min number of words: {min_num_words}")
        max_num_words = words.max()
        print(f"{dataset}: Max number of words: {max_num_words}")
        print("-----")
        print()



if __name__ == '__main__':
    if args.plot_locations:
        plot_locations(args)

    elif args.plot_similarity:
        plot_similarity(args)

    elif args.report_dataset_statistics:
        report_dataset_statistics(args)