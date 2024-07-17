

# how to prepare data
#collect instruction-response pairs
#concatenate pairs
#tokenize:pad,truncate
#split into train/test

# toknenize: 将数据转换成数字

import pandas as pd
import datasets
from pprint import pprint
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
text="how are you?"
encoded_text=tokenizer(text)["input_ids"]
print(encoded_text) #转换成的数字串

# tokenizer.decode(encoded_text) 调用失败

list_texts=["hi how are you?","i`m good","yes"]
encode_text=tokenizer(list_texts) #转换成数字

#处理的时候需要所有文本都是等长的,所以需要padding进行填充
tokenizer.pad_token=tokenizer.eos_token #没这玩意儿
encode_text=tokenizer(list_texts,padding=True) #转换成数字
#[12764,13,849,403,368,32],[42,1353,1175,0,0,0],[4374,0,0,0,0,0] padding为0,用0做了填充


tokenizer.truncation_side="left"
tokenizer_inputs=tokenizer(
    text,
    return_tensors="np",
    truncation=True,
    max_length=100,
)

def tokenize_function(examples):
    tokenizer.pad_token=tokenizer.eos_token
    token_inputs=tokenizer(
        text,
        return_tensors="np",
        padding=True,
    )
    max_length=min(token_inputs["input_ids"].shape[1],2048)
    tokenizer.truncation_side="left"
    return token_inputs

finetuning_dataset_loaded=datasets.load_dataset("json",data_files=filename)
tokenized_dataset=finetuning_dataset_loaded.map(
    tokenize_function,
    batched=True,
    batch_size=1,
    drop_last_batch=True
)
print(tokenized_dataset)
    # split the dataset
    tokenized_dataset=tokenized_dataset.add_column("labels",tokenized_dataset)
    split_dataset=tokenized_dataset.train_test_split(test_size=0.1,shuffle=True,seed=123)
    

if __name__ == '__main__':
    list_texts = ["hi how are you?", "i`m good", "yes"]
    encode_text = tokenizer(list_texts)  # 转换成数字