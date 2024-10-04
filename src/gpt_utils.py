from collections import defaultdict
from openai import OpenAI
from pandas import DataFrame
from tqdm import tqdm
import os
import pandas as pd
import json

openai_client = OpenAI()

def get_gpt_response_content(prompt, model, temperature, error_value):
  try:
    gpt_response = openai_client.chat.completions.create(
      messages = [{
        "role": "user", 
        "content": prompt
      }],
      model=model,
      temperature=temperature,
      max_tokens=2000
    )
    out = gpt_response.choices[0].message.content
    out = out.replace('```json', '')
    out = out.replace('```', '')

    left_brace_index = out.find('{')
    right_brace_index = out.find('}', left_brace_index)
    out = out[left_brace_index : right_brace_index + 1].strip()
    return out
  except (Exception, ValueError) as e:
    print(f"Exception: {e}")
    return error_value


def parse_gpt_response_content(gpt_response_content):
  return json.loads(gpt_response_content)
  

def process(df, 
            prompt_template, 
            var_col_dict, 
            var_val_dict,
            output_path,
            model='gpt-4-0613', 
            model_abrev='gpt4',
            temperature=0,
            force_rerun=False):
  if os.path.exists(output_path) and not force_rerun:
    gpt_results_df = pd.read_csv(output_path)
  else:
    gpt_results_df = _build_gpt_responses_data_df(df=df, 
                                                  prompt_template=prompt_template, 
                                                  var_col_dict=var_col_dict, 
                                                  var_val_dict=var_val_dict, 
                                                  model=model, 
                                                  temperature=temperature)
    gpt_results_df.to_csv(output_path, index=False)

  for col in gpt_results_df.columns:
    prefix = f'{model_abrev}_'
    mod_col = col.replace('gpt_', prefix)
    df[mod_col] = gpt_results_df[col]

  return df
  

def _build_gpt_responses_data_df(df: DataFrame,
                                 prompt_template: str, 
                                 var_col_dict,
                                 var_val_dict,
                                 model: str,
                                 temperature: int):
  gpt_results_dict = defaultdict(list)
  for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = prompt_template
    for var, col in var_col_dict.items():
      prompt = prompt.replace(var, str(row[col]))
    for var, val in var_val_dict.items():
      prompt = prompt.replace(var, val)
      
    gpt_response_content = get_gpt_response_content(prompt,
                                                    model=model,
                                                    temperature=temperature,
                                                    error_value="__ERROR__")
    
    gpt_response_content_dict = parse_gpt_response_content(gpt_response_content)

    for k, v in gpt_response_content_dict.items():
      gpt_results_dict[k].append(convert_yes_no_to_binary(v))
    
  gpt_results_df = DataFrame.from_dict(gpt_results_dict)

  return gpt_results_df


def convert_yes_no_to_binary(s):
  if s.lower() == 'no':
    return 0
  elif s.lower() == 'yes':
    return 1
  return s
