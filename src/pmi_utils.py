from tqdm import tqdm
from collections import defaultdict
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import open_coding_utils as oc
from constants import *
sns.set_theme(font='serif')

TEXTUAL_FEATURES = [
    'character_person',
    'event_experience',
    'plot_sequence',
    'setting_background',
    'literary_device',
    'fictional_hypothetical',
    'opinion',
    'behavior_strategy',
    'concept_definition',
    'artifact',
    'time'
]

EXTRA_TEXTUAL_FEATURES = [
    'problem_conflict',
    'theme_moral',
    'evocative_transporting',
    'cohesive_interpretable',
    'suspenseful',
    'creative',
    'feels_like_story',
    'implicitly_revealing',
    'emotion_sensation'
]

def preprocess(df, text_col_names, filter_to_keep=None):
    text_col_toks_list_dict = {k: [] for k in text_col_names}

    for text_col_name in text_col_names:
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            toks = row[text_col_name]
            if filter_to_keep == 'feature':
                toks = [tok for tok in toks if '1' in tok]
            if filter_to_keep == 'discourse':
                toks = [tok for tok in toks if '2' in tok]
            text_col_toks_list_dict[text_col_name].append(toks)

    return text_col_toks_list_dict


def get_unigram_count_dict(text_col_toks_list_dict, min_count=0):
    unigram_count_dict = defaultdict(int)
    for tok_group, toks_list in text_col_toks_list_dict.items():
        for toks in tqdm(toks_list, total=len(toks_list)):
            for tok in set(toks):
                unigram_count_dict[tok] += 1

    unigram_blocklist = set([key for key, value in unigram_count_dict.items() if value < min_count])

    unigram_count_dict = {k: v for k, v in unigram_count_dict.items() if k not in unigram_blocklist}

    return unigram_count_dict, unigram_blocklist


def filter_out_blocklisted_unigrams(text_col_toks_list_dict, unigram_blocklist):
    for tok_group, toks_list in text_col_toks_list_dict.items():
        toks_list = [[tok for tok in toks if tok not in unigram_blocklist] for toks in toks_list]
        text_col_toks_list_dict[tok_group] = toks_list

    return text_col_toks_list_dict


def get_cooccurrence_dict(text_col_toks_list_dict):
    coocurrence_dict = defaultdict(int)

    n = 0
    for tok_group, toks_list in text_col_toks_list_dict.items():
        n += len(toks_list)
        for toks in tqdm(toks_list, total=len(toks_list)):
            cooccurences_for_sentence = set()
            for i in range(len(toks[:-1])):
                for tok_j in toks[i + 1:]:
                    cooccurences_for_sentence.add(tuple(sorted((toks[i], tok_j))))

            for tup in cooccurences_for_sentence:
                coocurrence_dict[tup] += 1

    return coocurrence_dict, n


def get_pmi(word_1, word_2, unigram_count_dict, coocurrence_dict, N):
    tup = (word_1, word_2)
    cooccurence_count = coocurrence_dict[tuple(sorted(tup))]
    if cooccurence_count > 0:
        num = math.log2(
            (N * cooccurence_count) / (unigram_count_dict.get(word_1, 0) * unigram_count_dict.get(word_2, 0)))
        
        denom = - math.log2(cooccurence_count / N)
        return num / denom
    else:
        return 0


def execute(df, text_col_names, filename, min_unigram_count=0, min_cooccur=15, filter=None, plot_matrix=True):
    text_col_toks_list_dict = preprocess(df, text_col_names, filter)

    unigram_count_dict, unigram_blocklist = get_unigram_count_dict(text_col_toks_list_dict, min_unigram_count)

    text_col_toks_list_dict = filter_out_blocklisted_unigrams(text_col_toks_list_dict, unigram_blocklist)

    coocurrence_dict, n = get_cooccurrence_dict(text_col_toks_list_dict)
    print(coocurrence_dict.items())

    keys = unigram_count_dict.keys()

    pmi_scores = []

    heatmap_dict = defaultdict(list)
    for word_1 in keys:
        for word_2 in keys:
            pmi_score = get_pmi(word_1, word_2, unigram_count_dict, coocurrence_dict, n)
            tup = tuple(sorted((word_1, word_2)))
            if word_1 != word_2:
                tup_long_form = tuple(sorted((oc.decode(word_1), oc.decode(word_2))))
                tup_to_add = (round(pmi_score, 2), coocurrence_dict[tup], tup_long_form)
                if tup_to_add not in pmi_scores:
                    pmi_scores.append(tup_to_add)

            if coocurrence_dict[tup] >= min_cooccur:
                heatmap_dict[oc.decode(word_1)].append(pmi_score)
            else:
                heatmap_dict[oc.decode(word_1)].append(np.nan)
    
    pmi_scores = sorted(pmi_scores, key=lambda x: x[0], reverse=True)
    pmi_scores = [entry for entry in pmi_scores if entry[1] >= min_cooccur]
    print(pmi_scores)

    pmi_dict = defaultdict(list)
    main_col = 'Most and Least Co-occuring Feature Pairs'
    for entry in pmi_scores:
        pmi_dict[main_col].append(f'{entry[2][0]}, {entry[2][1]}')
        pmi_dict['PMI'].append(entry[0])
        pmi_dict['Count'].append(entry[1])
        f_1 = entry[2][0].replace('NOT_', '')
        f_2 = entry[2][1].replace('NOT_', '')
        pmi_dict['textual_and_extratextual'].append((f_1 in TEXTUAL_FEATURES and f_2 in EXTRA_TEXTUAL_FEATURES) or (f_1 in EXTRA_TEXTUAL_FEATURES and f_2 in TEXTUAL_FEATURES))

    pmi_df = pd.DataFrame(pmi_dict)

    if plot_matrix:
        heatmap_df = pd.DataFrame(heatmap_dict)
        heatmap_df.round(decimals=2)
        heatmap_df.index = [oc.decode(key) for key in keys]
        np.fill_diagonal(heatmap_df.values, np.nan)

        heatmap_df.replace(0, np.nan, inplace=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        heatmap = sns.heatmap(
            heatmap_df, 
            cmap="Blues", 
            annot=True, 
            cbar=False, 
            square=True, 
            annot_kws={"size": 7}
        )
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
        plt.subplots_adjust(bottom=0.2)

        plt.savefig(fname=f'{OUT_DIR}/{filename}.pdf', bbox_inches='tight', facecolor='white')
        plt.show()

    return pmi_df