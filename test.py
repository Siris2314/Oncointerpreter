from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
device = "cuda"
from langchain import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings



B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<s>>\n", "\n<</s>>\n\n"

def get_prompt(instruction, sys_prompt):
    system_prompt = B_SYS + sys_prompt + E_SYS
    template = B_INST + system_prompt +  instruction + E_INST
    return template



quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto", quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    use_cache=True,
    device_map = "auto",
    max_length = 500,
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)
llm = HuggingFacePipeline(pipeline=pipeline)

instruction = "Given the context that has been provided. \n {context}, Answer the following question: \n{question}"

sys_prompt = """You are a medical diagnosis expert.
You will be given context to answer from. Answer the questions with as much detail as possible and only in paragraphs.
In case you do not know the answer, you can say "I don't know" or "I don't understand".
In all other cases provide an answer to the best of your ability."""


prompt_sys = get_prompt(instruction, sys_prompt)


template = PromptTemplate(template=prompt_sys, input_variables=['context', 'question'])



