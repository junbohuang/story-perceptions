from llm2vec import LLM2Vec
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score


storyseeker_dataset = load_dataset("mariaantoniak/storyseeker")

train_df = pd.DataFrame(storyseeker_dataset['train'])
val_df = pd.DataFrame(storyseeker_dataset['val'])
test_df = pd.DataFrame(storyseeker_dataset['test'])

ids = train_df['id'].tolist() + val_df['id'].tolist() + test_df['id'].tolist()
reddit_df = pd.read_csv("../data/sp.csv")
id_text_dict = pd.Series(reddit_df.text.values,index=reddit_df.ids).to_dict()

train_df['text'] = train_df['id'].apply(lambda x: id_text_dict[x])
val_df['text'] = val_df['id'].apply(lambda x: id_text_dict[x])
test_df['text'] = test_df['id'].apply(lambda x: id_text_dict[x])

X_train = train_df['text'].tolist()
X_val = val_df['text'].tolist()
X_test = test_df['text'].tolist()

y_train = train_df['gold_consensus'].tolist()
y_val = val_df['gold_consensus'].tolist()
y_test = test_df['gold_consensus'].tolist()

# Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
tokenizer = AutoTokenizer.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
)
config = AutoConfig.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
model = PeftModel.from_pretrained(
    model,
    "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
)
model = model.merge_and_unload()  # This can take several minutes on cpu

# Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
model = PeftModel.from_pretrained(
    model, "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised"
)

# Wrapper for encoding and pooling operations
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

# train
max_iter = 100
batch_size = 8
scores = {}

clf = LogisticRegression(
    random_state=42,
    n_jobs=1,
    max_iter=max_iter,
    verbose=0,
)


X_train = np.asarray(l2v.encode(X_train, batch_size=batch_size))
y_train = np.asarray(y_train)

print("Fitting logistic regression classifier...")
clf.fit(X_train, y_train)
print("Evaluating...")
X_test = np.asarray(l2v.encode(X_test, batch_size=batch_size))
y_test = np.asarray(y_test)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
scores["accuracy"] = accuracy
f1 = f1_score(y_test, y_pred, average="macro")
scores["f1"] = f1

print(scores)