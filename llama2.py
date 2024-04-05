from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
device = "cuda"
from langchain import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader



B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_prompt(instruction, sys_prompt):
    system_prompt = B_SYS + sys_prompt + E_SYS
    template = B_INST + system_prompt +  instruction + E_INST
    return template



quantization_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto", quantization_config=quantization_config, token="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")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    use_cache=True,
    device_map = "auto",
    max_length = 2048,
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


from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.vectorstores import Chroma
import nest_asyncio

import textwrap


def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    
    wrapped_text = '\n'.join(wrapped_lines)
    
    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['text']))
    print("\n\nSources:")
    for source in llm_response['context']:
        print(source.metadata['source'])




nest_asyncio.apply()

articles = ["https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118",
            "https://www.webmd.com/heart-disease/heart-disease-types-causes-symptoms",
            "https://www.cdc.gov/heartdisease/facts.htm",
            "https://www.healthline.com/health/heart-disease/causes-risks",
            "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)"]

# Scrapes the blogs above
loader = AsyncChromiumLoader(articles)
docs = loader.load()


from langchain.schema.runnable import RunnablePassthrough


# Converts HTML to plain text 
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

# Chunk text
text_splitter = CharacterTextSplitter(chunk_size=500, 
                                      chunk_overlap=0)
chunked_documents = text_splitter.split_documents(docs_transformed)

# Load chunked documents into the FAISS index
db = Chroma.from_documents(chunked_documents, 
                          HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en",
                                        model_kwargs={'device': 'cuda'}, encode_kwargs={'normalize_embeddings': True}))


# Connect query to FAISS index using a retriever
# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={'k': 4}
# )

# query = "What are some causes heart disease?"
# docs = db.similarity_search(query)
# print(docs[0].page_content)

llm_chain = LLMChain(llm=llm, prompt=template)

# print(llm_chain.invoke({"context":"", 
#                   "question": "What causes heart disease?"}))


query = "Is high blood pressure and heart disease related?" 

retriever = db.as_retriever()

rag_chain = ( 
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)

ans = rag_chain.invoke(query)
process_llm_response(ans)