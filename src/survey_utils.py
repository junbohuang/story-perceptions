import math
import numpy as np
import pandas as pd


def get_likert_score(row, keyword):
    for i in range(1, 6):
        score = row[f'{keyword}:::scale_{i}']
        if not math.isnan(score):
            return i
    return np.NaN


def get_confidence_score(row):
    return get_likert_score(row, 'confidence')


def get_familiarity_score(row):
    return get_likert_score(row, 'familiarity')


def get_label(row):
    if not pd.isnull(row['story_decision:::NO']) and row['story_decision:::NO'] == 2.0:
        return 0
    elif not pd.isnull(row['story_decision:::YES']) and row['story_decision:::YES'] == 1.0:
        return 1
    else:
        return np.NaN


def get_code_list(codes_str):
    return str(codes_str).strip('[]').split() if type(codes_str) is str else []


def instance_coded(prolific_mod_row):
    # require either goal_codes or rationale_codes (tolerate edge case where no codes applied to one of these questions)
    return prolific_mod_row['goal_codes'] != [] or prolific_mod_row['rationale_codes'] != []