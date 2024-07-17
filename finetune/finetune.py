from lamini import BasicModelRunner
from llama import LlamaV3Runner
import lamini
import warnings




if __name__ == '__main__':
    warnings.filterwarnings(action="ignore")
    lamini.api_key = "e87d838ed15fb9750a0ad99e1b8a06387bfa6c0a6a746595b31d9610642c74c9"

    # non_finetuned = lamini.Lamini("meta-llama/Meta-Llama-3-8B-Instruct")
    # print(llm.generate("How are you?"))
    non_finetuned = lamini.Lamini("meta-llama/Llama-2-7b-hf")
    non_finetuned_output = non_finetuned.generate("tell me how to train my dog to sit")
    print(non_finetuned_output)



