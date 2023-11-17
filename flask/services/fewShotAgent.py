from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
import subprocess                           
from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import TextLoader
from langchain.agents import initialize_agent
from langchain.prompts import FewShotPromptTemplate
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import DirectoryLoader
import os
import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter


# import sys

def main(query):
    print(query)
    load_dotenv()
    OPEN_AI_API_KEY = os.environ.get('OPEN_AI_API_KEY')

    print(os.getcwd())
    file_path = os.path.join(os.getcwd(), "data/subtitle.txt")
    try:
        with open(file_path) as f:
            transcript = f.read()
        print(type(transcript))
    except Exception as e:
        print(e)
    print("trancript from filez:")
   
    loader = UnstructuredFileLoader(os.getcwd()+"/data/subtitle.txt",silent_errors=True,show_progress=True,)

    doc= loader.load()


    print(transcript)
    try:
        text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 5000,
    chunk_overlap  = 250,
    length_function = len,
    is_separator_regex = False,
    )
        partial_transcripts = text_splitter.split_documents(documents=doc)
        
    except Exception as e:
        print(e)

    
    try:
               
        # llm = Ollama(model="mistral", temperature=0,verbose=True,num_gpu=1,num_ctx=8000,)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo",openai_api_key=OPEN_AI_API_KEY,verbose=True)
        print("LLM ready!")
        

    except Exception as e:
        print(f"e :{e}")

    prompt_file_path = os.getcwd()+"/flask/services/init_prompt.txt"
    temp=""
    with open(prompt_file_path, 'r', encoding='utf-8') as file:
        temp=file.read()
    prompt_template= temp
    prompt = PromptTemplate.from_template(prompt_template)
    prompt_file_path = os.getcwd()+"/flask/services/refine_prompt.txt"
    with open(prompt_file_path, 'r', encoding='utf-8') as file:
        temp=file.read()
    refine_template = temp
    refine_prompt = PromptTemplate.from_template(refine_template)
    
    try:
        
        print("chain summarize loading")
        chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        verbose=True ,
        input_key="input_documents",
        output_key="output_text",
        )
        print("chain summarize loading done")
        result = chain({"input_documents": partial_transcripts}, return_only_outputs=True)
        print("result ::\n\n\n\n\n")

        print(result["output_text"])
    except Exception as e:
        print(e)
    return "notes generated"


