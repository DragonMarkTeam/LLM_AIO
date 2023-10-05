from huggingface_hub import login
from datasets import load_dataset
import datasets
from datasets import Dataset, DatasetDict
import pandas as pd
from datasets import load_dataset
login()


def find_true_label(x):
  text = x["choices.text"]
  for i in range(len(text)):
    if x["answerKey"]	== x["choices.label"][i]:
      return text[i]

def load_data(path, cache_dir, splits):
    print(splits)
    multichoices = False
    if path == "uonlp/CulturaX":
        ds = load_dataset(path,
                        language="vi",
                        split=splits,
                        use_auth_token=True)
        drop_columns = ["id", "choices.label","answerKey"]
        dict = {"question": "input",
                "choices.text": "extra_input"}
        multichoices=True
    elif path == "Anthropic/hh-rlhf":
        ds = load_dataset(path,
                        split=splits,
                        use_auth_token=True)
        drop_columns = []
        dict = {"rejected": "input",
        "chosen": "label"}
    elif path == "Open-Orca/OpenOrca":
        ds = load_dataset(path,
                        split=splits,
                        use_auth_token=True)
        drop_columns = ["id", "system_prompt"]
        dict = {"question": "input",
                "response": "label"}
    elif path == "tatsu-lab/alpaca":
        ds = load_dataset(path,
                        split=splits,
                        use_auth_token=True)
        drop_columns = ["text"]
        dict = {"instruction": "input",
                "input": "extra_input",
                "output": "label"}
    elif path == "vlsp-2023-vllm/lambada_vi":
        ds = load_dataset(path,
                        split=splits,
                        use_auth_token=True)
        drop_columns = ["text",'metadata.num_sents',
       'metadata.target_word.appeared_in_prev_sents',
       'metadata.target_word.pos_tag', 'metadata.title', 'metadata.url',
       'metadata.word_type']
        dict = {"context": "input",
                "target_word": "label"}
    elif path == "vlsp-2023-vllm/grade_12_exams":
        ds = load_dataset(path,
                        split=splits,
                        use_auth_token=True)
        drop_columns = ["id", "choices.label", "answerKey", 'metadata.grade', 'metadata.language', 'metadata.subject']
        dict = {"question": "input",
                "choices.text": "extra_input"}
        multichoices=True
    elif path == "vlsp-2023-vllm/ai2_arc_vi":
        ds = load_dataset(path,
                        cache_dir=cache_dir,
                        split=splits,
                        use_auth_token=True)
        drop_columns = ["id", "choices.label", "answerKey"]
        dict = {"question": "input",
                "choices.text": "extra_input"}
        multichoices=True

        
    data = pd.json_normalize(ds)
    if multichoices == True:
        data["label"] = data.apply(lambda x: find_true_label(x), axis=1)
    if "extra_input" not in data.columns:
        data["extra_input"] = " "
    data.drop(drop_columns, axis='columns', inplace=True)
    data.rename(columns=dict, inplace=True)
    df = Dataset.from_pandas(data)
    
    dataset = DatasetDict()
    dataset[splits] = df
    return dataset