import jsonlines


from lamini import BasicModelRunner

def instruction():
    prompt_template_with_input="""
    below is an instruction that describes a task
    ### Instruction:
    {instruction}
    ### input:
    {input}
    ### Response:
    """

    prompt_template_without_input="""
    below is an instruction that describes a task
    ### Instruction:
    {instruction}
    ### Response:
    """

    # processed_data=[]
    # with open("","rb") as f:
    #     data=f.read()
    #     filename=f.name
    # #拆分数据集,如果数据库单项里面有key为input的
    # if not xx["input"]:
    #     processed_prompt=prompt_template_without_input.format(instruction=xx["instruction"],input=xx["input"])
    # else:
    #     processed_prompt = prompt_template_without_input.format(instruction=xx["instruction"])
    # processed_data.append({"input":processed_prompt,"output":xx["output"]})
    #
    # with jsonlines.open("dataset.jsonl","w") as f:
    #     f.write_all(processed_data)



    instruct-model=BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")  #meta-llama/Llama-2-7b-chat-hf是inst调试过的,meta-llama/Llama-2-7b-hf是没调试过的,所以这里进行了对比
    resp=instruct-model("tell me how to train my dog")

    # inference函数
