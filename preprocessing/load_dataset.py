from huggingface_hub import login
login()

from datasets import load_dataset
ds = load_dataset("uonlp/CulturaX",
                  language="vi",
                  use_auth_token=True)
