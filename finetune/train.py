from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch

def inference(text,model,tokenizer,max_input_tokens=1000,max_out_token=2048):
    #tokenize
    input_ids=tokenizer.encode(
        text,return_tensors="pt",
        truncation=True,max_length=max_input_tokens
    )
    #generate
    device=model.device
    generated_token_with_prompt=model.generate(
        input_ids=input_ids.to(devices),
        max_length=max_out_token
    )
    #decode
    generated_text_with_prompt=tokenizer.batch_decode(generated_token_with_prompt)
    #strip the prompt
    generated_text_answer=generated_text_with_prompt[0][len(text):]
    return generated_text_answer


# train
# 遍历dataset,分批将dataset放入model中,从模型计算loss损失,反向传播,更新优化器
dataset_name="lamini_docs.jsonl"
dataset_path=f"/content/{dataset_name}"
use_hf=False
model_name="EleutherAI/pythia-70m"
training_config={
    "model":{
        "pretrained_name":model_name,
        "max_lenght":2048
    },
    "datasets":{
        "use_hf":use_hf,
        "path":dataset_path
    },
    "verbose":True
}

tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
train_dataset,test_dateset=tokenize_and_split(training_config,tokenizer)

base_model=AutoModelForCausalLM.from_pretrained(model_name)
device_count=torch.cuda.device_count()
if device_count>0:
    device=torch.device("cuda") #gpu
else:
    device=torch.device("cpu") #cpu

base_model.to(device) #python没给出返回类型,所以方法不提示
# AutoModelForCausalLM.from_pretrained(
#     'THUDM/cogvlm-chat-hf',
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True
# ).to('cuda').eval()

print(inference(test_dateset[0]['question'],base_model,tokenizer))
