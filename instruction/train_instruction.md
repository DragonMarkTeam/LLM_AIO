# Mask random
https://huggingface.co/learn/nlp-course/chapter7/3?fw=tf&fbclid=IwAR0lobymgLmOeNlmB1nu5y3qp7cqY_8U22M0lB7dsSbk1T3yOU-sS_9zvHY

Cái này có tên là datacollection

'''
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
'''

Sau đây là code ví dụ 

'''
samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")

for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
'''