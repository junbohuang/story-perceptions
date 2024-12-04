import pandas as pd
from constants import *
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from zipfile import ZipFile
import json
import os
import time
import io
from zipfile import ZipFile
import requests
from tqdm import tqdm
import json
import llama_utils
import gpt_utils
from llm_prompt_templates import *
import survey_utils
from collections import defaultdict
import metric_utils
import pandas_utils

ss_dataset_dict = load_dataset(STORY_SEEKER_DATASET_NAME)
ss_split_dfs = [pd.DataFrame(ss_dataset_dict[split]) for split in ss_dataset_dict.keys()]
ss_df = pd.concat(ss_split_dfs, ignore_index=True)

def _format_size(size_bytes):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def download_with_progress(url, chunk_size=8192):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    start_time = time.time()
    downloaded_size = 0
    data = io.BytesIO()
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = data.write(chunk)
            downloaded_size += size
            pbar.update(size)
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                speed = downloaded_size / elapsed_time
                pbar.set_postfix(speed=f"{_format_size(speed)}/s", refresh=True)
    return data

if not os.path.exists(TLDR_SS_SUBSET_PATH) or FORCE_REHYDRATE_TLDR:
    ss_ids = set(ss_df['id'].values)
    tldr_ss_subset_dict = defaultdict(dict)

    print("Starting download...")
    tldr_zip_data = download_with_progress(TLDR_URL)

    print("Processing TLDR_17 ZIP file contents...")
    with ZipFile(tldr_zip_data, 'r') as zip_obj:
        print("ZIP file opened successfully")
        for _filename in zip_obj.namelist():
            with zip_obj.open(_filename) as _file:
                for _line in tqdm(_file):
                    _data = json.loads(_line)
                    if _data['id'] in ss_ids:
                        tldr_ss_subset_dict[_data['id']] = _data

    print("Saving TLDR_17 instances that intersect with StorySeeker...")
    tldr_ss_subset_df = pd.DataFrame.from_dict(tldr_ss_subset_dict, orient='index')
    tldr_ss_subset_df.to_csv(TLDR_SS_SUBSET_PATH, index=False)
else:
    tldr_ss_subset_df = pd.read_csv(TLDR_SS_SUBSET_PATH)

# data is small enough to work with in dictionary format for convenience
tldr_ss_subset_dict = tldr_ss_subset_df.set_index('id').T.to_dict()
ss_df['text'] = ss_df['id'].apply(lambda id: tldr_ss_subset_dict[id]['body'])
# ss_df.to_csv(SS_PATH, index=False)


pcr_df = pd.read_csv(POTATO_CODED_RAW_FILTERED_PATH, converters=pandas_utils.get_list_converters(POTATO_CODED_RAW_FILTERED_PATH))
print('pcr_df count:', len(pcr_df.index))
sse_df = ss_df.copy()

for model_abrev, model in [('llama3', ''),
                           ('gpt4o', 'gpt-4o-2024-05-13'),
                           ('gpt4t', 'gpt-4-turbo-2024-04-09'),
                           ('gpt4', 'gpt-4-0613')]:
    for prompt_idx, prompt_template in enumerate([TEXT_TO_DECISION_AND_RATIONALE_0,
                                                  TEXT_TO_DECISION_AND_RATIONALE_1,
                                                  TEXT_TO_DECISION_AND_RATIONALE_2,
                                                  TEXT_TO_DECISION_AND_RATIONALE_3,
                                                  TEXT_TO_DECISION_AND_RATIONALE_4]):
        if 'gpt' in model_abrev and (
                f'{model_abrev}_descriptive_label_{prompt_idx}' not in sse_df.columns or FORCE_RERUN_GPT):
            output_path = f'{GPT_RESULTS_DIR}/{model_abrev}/text_to_decision_and_rationale_{prompt_idx}.csv'
            sse_df = gpt_utils.process(df=sse_df,
                                       prompt_template=prompt_template,
                                       var_col_dict={"[TEXT]": 'text'},
                                       var_val_dict={},
                                       output_path=output_path,
                                       force_rerun=FORCE_RERUN_GPT,
                                       model=model,
                                       model_abrev=model_abrev)
            sse_df.to_csv(SS_EXTENDED_PATH, index=False)

        elif 'llama3' in model_abrev and (
                f'{model_abrev}_descriptive_label_{prompt_idx}' not in sse_df.columns or FORCE_RERUN_LLAMA3):
            output_path = f'{LLAMA3_RESULTS_DIR}/text_to_decision_and_rationale_{prompt_idx}.csv'
            sse_df = llama_utils.process(df=sse_df,
                                         prompt_template=prompt_template,
                                         var_col_dict={"[TEXT]": 'text'},
                                         var_val_dict={},
                                         output_path=output_path,
                                         idx=prompt_idx,
                                         force_rerun=FORCE_RERUN_LLAMA3,
                                         model_abrev=model_abrev)
            sse_df.to_csv(SS_EXTENDED_PATH, index=False)

    sse_df = pd.read_csv(SS_EXTENDED_PATH)

    cols = [f'{model_abrev}_descriptive_label_{prompt_idx}' for prompt_idx in range(5)]
    sse_df[f'{model_abrev}_descriptive_label_mv'] = sse_df[cols].mode(axis=1)[0]
    sse_df[f'{model_abrev}_descriptive_label_union'] = sse_df[cols].max(axis=1)

    sse_df.to_csv(SS_EXTENDED_PATH, index=False)


pc_df = pcr_df.copy()

pc_df['confidence'] = pc_df.apply(lambda row: survey_utils.get_confidence_score(row), axis=1)
pc_df['familiarity'] = pc_df.apply(lambda row: survey_utils.get_familiarity_score(row), axis=1)
pc_df['label'] = pc_df.apply(lambda row: survey_utils.get_label(row), axis=1)
pc_df['gc_label'] = pc_df['instance_id'].apply(lambda instance_id: sse_df.loc[sse_df['id'] == instance_id, 'gold_consensus'].iloc[0])
pc_df['gpt4_descriptive_label_mv'] = pc_df['instance_id'].apply(lambda instance_id: sse_df.loc[sse_df['id'] == instance_id, 'gpt4_descriptive_label_mv'].iloc[0])
pc_df['gpt4t_descriptive_label_mv'] = pc_df['instance_id'].apply(lambda instance_id: sse_df.loc[sse_df['id'] == instance_id, 'gpt4t_descriptive_label_mv'].iloc[0])
pc_df['gpt4o_descriptive_label_mv'] = pc_df['instance_id'].apply(lambda instance_id: sse_df.loc[sse_df['id'] == instance_id, 'gpt4o_descriptive_label_mv'].iloc[0])
pc_df['llama3_descriptive_label_mv'] = pc_df['instance_id'].apply(lambda instance_id: sse_df.loc[sse_df['id'] == instance_id, 'llama3_descriptive_label_mv'].iloc[0])

pc_df['goal'] = pc_df['goal:::text_box']
pc_df['goal_codes'] = pc_df['goal_codes'].apply(survey_utils.get_code_list)
pc_df['rationale'] = pc_df['story_decision_explanation:::text_box']
pc_df['rationale_codes'] = pc_df['story_decision_explanation_codes'].apply(survey_utils.get_code_list)
pc_df['alternative'] = pc_df['story_alternative:::text_box']
pc_df['alternative_codes'] = pc_df['story_alternative_codes'].apply(survey_utils.get_code_list)
pc_df['is_coded'] = pc_df.apply(lambda row: survey_utils.instance_coded(row), axis=1)
pc_df = pc_df[['user', 'instance_id', 'confidence', 'familiarity', 'label', 'gc_label', 'gpt4_descriptive_label_mv', 'gpt4t_descriptive_label_mv', 'gpt4o_descriptive_label_mv', 'llama3_descriptive_label_mv', 'goal', 'goal_codes', 'rationale', 'rationale_codes', 'alternative', 'alternative_codes', 'is_coded']]

pc_df = pc_df[pc_df['is_coded'] == True]
pc_df.to_csv(POTATO_CODED_PATH, index=False)


default_dict_for_instance = lambda: defaultdict(list)
instance_crowd_dict = defaultdict(default_dict_for_instance)
for i, row in pc_df.iterrows():
    instance_id = row['instance_id']
    confidence = row['confidence']
    familiarity = row['familiarity']
    label = row['label']
    instance_crowd_dict[instance_id]['crowd_confidence_scores'].append(confidence)
    instance_crowd_dict[instance_id]['crowd_familiarity_scores'].append(familiarity)
    instance_crowd_dict[instance_id]['crowd_labels'].append(label)
    instance_crowd_dict[instance_id]['crowd_goals'].append(row['goal'])

instance_crowd_df = pd.DataFrame.from_dict(instance_crowd_dict).transpose()
instance_crowd_df['crowd_confidence_avg'] = instance_crowd_df['crowd_confidence_scores'].apply(lambda x: metric_utils.get_avg(x))
instance_crowd_df['crowd_familiarity_avg'] = instance_crowd_df['crowd_familiarity_scores'].apply(lambda x: metric_utils.get_avg(x))
instance_crowd_df['crowd_label_union'] = instance_crowd_df['crowd_labels'].apply(lambda x: 1 if 1 in x else 0)
instance_crowd_df['crowd_label_mv'] = instance_crowd_df['crowd_labels'].apply(lambda x: metric_utils.get_majority_vote(x))
instance_crowd_df['crowd_label_mv_rate'] = instance_crowd_df['crowd_labels'].apply(lambda x: metric_utils.get_majority_vote_rate(x))
instance_crowd_df['crowd_confidence_entropy'] = instance_crowd_df['crowd_confidence_scores'].apply(metric_utils.get_entropy)
instance_crowd_df['crowd_familiarity_entropy'] = instance_crowd_df['crowd_familiarity_scores'].apply(metric_utils.get_entropy)
instance_crowd_df['crowd_labels_entropy'] = instance_crowd_df['crowd_labels'].apply(metric_utils.get_entropy)

sp_df = pd.merge(sse_df, instance_crowd_df, how='left', left_on='id', right_index=True)
sp_df.to_csv(SP_PATH)
print('sp_df count:', len(sp_df.index))


pc_df['crowd_confidence_avg'] = pc_df['instance_id'].apply(lambda instance_id: sp_df.loc[sp_df['id'] == instance_id, 'crowd_confidence_avg'].iloc[0])
pc_df['crowd_label_mv'] = pc_df['instance_id'].apply(lambda instance_id: sp_df.loc[sp_df['id'] == instance_id, 'crowd_label_mv'].iloc[0])
pc_df['crowd_label_mv_rate'] = pc_df['instance_id'].apply(lambda instance_id: sp_df.loc[sp_df['id'] == instance_id, 'crowd_label_mv_rate'].iloc[0])
pc_df['crowd_labels_entropy'] = pc_df['instance_id'].apply(lambda instance_id: sp_df.loc[sp_df['id'] == instance_id, 'crowd_labels_entropy'].iloc[0])
pc_df['crowd_confidence_entropy'] = pc_df['instance_id'].apply(lambda instance_id: sp_df.loc[sp_df['id'] == instance_id, 'crowd_confidence_entropy'].iloc[0])
pc_df.to_csv(POTATO_CODED_PATH)