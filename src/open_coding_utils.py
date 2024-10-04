from collections import defaultdict
import pandas as pd
from constants import *
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import get_font_names
sns.set_theme(font='serif')

codes_df = pd.read_csv(CODES_PATH)
code_id_short_name_dict = {row[1]: row[2] for _, row in codes_df.iterrows()}

def decode(code):
    prefix = ''
    suffix = code

    if code.startswith('!*'):
        prefix = "NOT_VERB_"
        suffix = code[2:]
    elif code.startswith('!'):
        prefix = "NOT_"
        suffix = code[1:]
    elif code.startswith('*'):
        prefix = "VERB_"
        suffix = code[1:]

    suffix = code_id_short_name_dict[suffix] if suffix in code_id_short_name_dict else suffix

    return prefix + suffix

def hard_filter_codes(codes, type_to_keep, ignore_discourse_nouns=False):
    if type_to_keep == 'feature':
        codes = [code for code in codes if '1' in code]
    elif type_to_keep == 'discourse':
        codes = [code for code in codes if '2' in code]

    if ignore_discourse_nouns:
        codes = [code for code in codes if '*' in code]

    return codes


def get_pos_agnostic_codes(codes):
    return [code.replace("*", "") for code in codes]


def apply_all_filters(codes,
                      hard_filter_type_to_keep=None,
                      ignore_discourse_nouns=False):
    codes = codes.copy()
    filtered_code_list = hard_filter_codes(codes=codes,
                                           type_to_keep=hard_filter_type_to_keep,
                                           ignore_discourse_nouns=ignore_discourse_nouns)
    return filtered_code_list


def get_code_names_and_counts(df,
                              col_names,
                              hard_filter_type_to_keep=None,
                              ignore_discourse_nouns=False):
    code_count_dict = defaultdict(int)
    for i, row in df.iterrows():
        codes_seen = set()
        filtered_code_list = apply_all_filters(codes=row[col_names[0]],
                                               hard_filter_type_to_keep=hard_filter_type_to_keep,
                                               ignore_discourse_nouns=ignore_discourse_nouns)
        if len(col_names) > 1:
            for col_name in col_names[1:]:
                filtered_code_list.extend(
                    apply_all_filters(codes=row[col_name],
                                      hard_filter_type_to_keep=hard_filter_type_to_keep,
                                      ignore_discourse_nouns=ignore_discourse_nouns)
                )

        for code in filtered_code_list:
            if code not in codes_seen:
                codes_seen.add(code)
                code_count_dict[code] += 1

    return code_count_dict


def get_code_proportions_dict(code_counts_dict, size):
    code_proportions_dict = code_counts_dict.copy()
    total_code_count = sum(code_counts_dict.values())
    for key, value in code_counts_dict.items():
        code_proportions_dict[key] = value / size
    return code_proportions_dict


def decode_and_sort_code_counts(code_count_dict):
    sorted_code_count_items = sorted(code_count_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_code_names, sorted_code_counts = zip(*sorted_code_count_items)
    sorted_decoded_code_names = [decode(code) for code in sorted_code_names]
    return sorted_decoded_code_names, sorted_code_counts


def calculate_diff_counts(y_code_counts, n_code_counts, min_code_count=50):
    observed_codes = {*y_code_counts.keys(), *n_code_counts.keys()}
    diff_dict = defaultdict(int)
    for observed_code in observed_codes:
        diff_dict[observed_code] = y_code_counts[observed_code] - n_code_counts[observed_code]

    return diff_dict


def apply_min_code_count_filter(y_code_counts, n_code_counts, y_code_props, n_code_props, min_code_count=50):
    observed_codes = {*y_code_counts.keys(), *n_code_counts.keys()}
    for observed_code in observed_codes:
        if y_code_counts[observed_code] + n_code_counts[observed_code] < min_code_count:
            y_code_props.pop(observed_code, None)
            n_code_props.pop(observed_code, None)

    return y_code_props, n_code_props


def apply_min_code_count_filter_simple(code_counts, code_props, min_code_count=50):
    for observed_code in code_counts:
        if code_counts[observed_code] < min_code_count:
            code_props.pop(observed_code, None)

    return code_props

def plot_proportion_diffs(group1_code_counts_dict, group1_size, group2_code_counts_dict, group2_size, out_filename='test', min_code_count=50, group1_name='group1', group2_name='group2', y_axis_label='Codes', arrow_caption='', threshold=0.03):
    group1_code_proportions_dict = get_code_proportions_dict(group1_code_counts_dict, group1_size)
    group2_code_proportions_dict = get_code_proportions_dict(group2_code_counts_dict, group2_size)
    group1_code_proportions_dict, group2_code_proportions_dict = apply_min_code_count_filter(group1_code_counts_dict,
                                                                                             group2_code_counts_dict,
                                                                                             group1_code_proportions_dict,
                                                                                             group2_code_proportions_dict,
                                                                                             min_code_count)
    group_diff_code_proportions_dict = calculate_diff_counts(group1_code_proportions_dict,
                                                             group2_code_proportions_dict,
                                                             min_code_count)
    
    print(group_diff_code_proportions_dict.items())
    group_diff_code_proportions_dict = {k:v for k,v in group_diff_code_proportions_dict.items() if abs(v) >= threshold}
    # print(group_diff_code_proportions_dict.items())
    sorted_code_names, sorted_code_proportions = decode_and_sort_code_counts(group_diff_code_proportions_dict)

    num_bars = len(sorted_code_names)
    fig_width = 6
    fig_height = 1.2 + num_bars * 0.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    is_negative_code = ['NOT_' in code_name for code_name in sorted_code_names]
    palette = ['C0', 'C1']  # Colors for negative and non-negative codes
    sns.barplot(x=sorted_code_proportions, y=sorted_code_names, hue=is_negative_code, palette=palette, ax=ax, legend=False)
    
    xlabel_prefix = '' if arrow_caption == '' else f'{arrow_caption}\n'
    ax.set_xlabel(f'{xlabel_prefix}Prob Diff ({group1_name} - {group2_name})', fontsize=12)
    ax.set_ylabel(y_axis_label, fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fname=f'{OUT_DIR}/{out_filename}.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

    return sorted_code_names, sorted_code_proportions 

def plot_props(group1_code_counts_dict, group1_size, out_filename='test', min_code_count=50, group1_name='group1', y_axis_label='Codes'):
    group1_code_proportions_dict = get_code_proportions_dict(group1_code_counts_dict, group1_size)

    group1_code_proportions_dict = apply_min_code_count_filter_simple(group1_code_counts_dict, 
                                                                      group1_code_proportions_dict,
                                                                      min_code_count)
    
    sorted_code_names, sorted_code_proportions = decode_and_sort_code_counts(group1_code_proportions_dict)

    num_bars = len(sorted_code_names)
    fig_width = 6
    fig_height = 0.75 + num_bars * 0.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    is_negative_code = ['NOT_' in code_name for code_name in sorted_code_names]
    palette = ['C0', 'C1']  # Colors for negative and non-negative codes
    sns.barplot(x=sorted_code_proportions, y=sorted_code_names, hue=is_negative_code, palette=palette, ax=ax, legend=False)
    ax.set_xlabel(f'Prob ({group1_name})', fontsize=12)
    ax.set_ylabel(y_axis_label, fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fname=f'{OUT_DIR}/{out_filename}.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

def plot_counts(code_counts_dict, out_filename='test', min_code_count=50, y_axis_label='Codes'):
    code_counts_dict = {code: count for code, count in code_counts_dict.items() if count > min_code_count}
    sorted_code_names, sorted_code_proportions = decode_and_sort_code_counts(code_counts_dict)

    num_bars = len(sorted_code_names)
    fig_width = 6
    fig_height = 0.75 + num_bars * 0.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    is_negative_code = ['NOT_' in code_name for code_name in sorted_code_names]
    palette = ['C0', 'C1']  # Colors for negative and non-negative codes
    sns.barplot(x=sorted_code_proportions, y=sorted_code_names, hue=is_negative_code, palette=palette, ax=ax, legend=False)
    ax.set_xlabel(f'Counts', fontsize=12)
    ax.set_ylabel(y_axis_label, fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fname=f'{OUT_DIR}/{out_filename}.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

def remove_all_but_primary_discourse_code(codes):
  for code in codes:
    if '2' in code:
      code = code.replace("*", "")
      return [code]
  return []