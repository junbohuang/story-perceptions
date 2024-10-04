import transformers
import torch
import re
from collections import defaultdict
from pandas import DataFrame
from tqdm import tqdm
import os
import pandas as pd
import json
from pandas import DataFrame
from tqdm import tqdm
import os
import pandas as pd
from passwords import hf_token

model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token=hf_token
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def get_llama3_response_content(prompt, error_value, idx, model_abrev):
  messages = [
      {"role": "system", "content": "You write valid json with open and closing braces. When asked to provide a label for whether a text contains or lacks a story, you must respond with either 'yes' or 'no'."},
      {"role": "user", "content": prompt},
  ]
  
  try:
    llama3_response = pipeline(
        messages,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=False,
        pad_token_id=pipeline.tokenizer.eos_token_id
    )

    out = llama3_response[0]["generated_text"][-1]['content']

    left_brace_index = out.find('{')
    right_brace_index = out.find('}', left_brace_index)
    if right_brace_index != -1:
      out = out[left_brace_index : right_brace_index + 1].strip()
    else:
      out = out[left_brace_index :].strip()

    # Using regex to extract key-value pairs
    pattern = r'"(.*?)":\s*"(.*?)"'
    matches = re.findall(pattern, out)
    
    label = matches[0][1]
    rationale = matches[1][1] if len(matches) > 1 else ''
    answers = {
      f"{model_abrev}_descriptive_label_{idx}": label,
      f"{model_abrev}_descriptive_label_rationale_{idx}": rationale 
    }
    return json.dumps(answers)
  except (Exception, ValueError) as e:
    print(f"Exception: {e}")
    raise e

def parse_llama3_response_content(llama3_response_content):
  return json.loads(llama3_response_content)
  
def process(df, 
            prompt_template, 
            var_col_dict, 
            var_val_dict,
            output_path,
            idx,
            model_abrev='llama3',
            force_rerun=False):
  if os.path.exists(output_path) and not force_rerun:
    llama3_results_df = pd.read_csv(output_path)
  else:
    llama3_results_df = _build_llama3_responses_data_df(df=df, 
                                                        prompt_template=prompt_template, 
                                                        var_col_dict=var_col_dict, 
                                                        var_val_dict=var_val_dict,
                                                        idx=idx,
                                                        model_abrev=model_abrev)
    llama3_results_df.to_csv(output_path, index=False)

  for col in llama3_results_df.columns:
    df[col] = llama3_results_df[col]

  return df
  

def _build_llama3_responses_data_df(df: DataFrame,
                                 prompt_template: str, 
                                 var_col_dict,
                                 var_val_dict,
                                 idx,
                                 model_abrev):
  llama3_results_dict = defaultdict(list)
  for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = prompt_template
    for var, col in var_col_dict.items():
      prompt = prompt.replace(var, str(row[col]))
    for var, val in var_val_dict.items():
      prompt = prompt.replace(var, val)

    llama3_response_content = get_llama3_response_content(prompt,
                                                          error_value="__ERROR__", 
                                                          idx=idx, 
                                                          model_abrev=model_abrev)
    
    llama3_response_content_dict = parse_llama3_response_content(llama3_response_content)

    for k, v in llama3_response_content_dict.items():
      llama3_results_dict[k].append(convert_yes_no_to_binary(v))
    
  llama3_results_df = DataFrame.from_dict(llama3_results_dict)
  return llama3_results_df

def convert_yes_no_to_binary(s):
  if s.lower() == 'no':
    return 0
  elif s.lower() == 'yes':
    return 1
  return s
