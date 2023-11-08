from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
                                 


from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import TextLoader
from langchain.agents import initialize_agent
from langchain import FewShotPromptTemplate
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.agents import Tool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import DirectoryLoader
import os
import json
# import sys

def main(query):
    print(query)
    load_dotenv()
    OPEN_AI_API_KEY = os.environ.get('OPEN_AI_API_KEY')


    loader = DirectoryLoader(os.getcwd()+"/data/output_chunks/", glob="**/*.txt",silent_errors=True,show_progress=True,use_multithreading=True)
    # loader = TextLoader(os.getcwd()+"/services/data/output_chunks/chunk_1.txt")
    # loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    print("directory loaded")
    docs = loader.load()
    print(docs.count())
    print("text loaded")
    llm = Ollama(model="mistral", 
             temperature=0.3,verbose=True)
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo",openai_api_key=OPEN_AI_API_KEY)
    print("LLM ready!")
    chain = load_summarize_chain(llm, chain_type="refine")
    chain.run(docs)

    print("refining")


#     prompt_template = """Write structured notes in markdown (.md) format which is compatible with Obsidian note taking app of the following and You will adhere to this given template  :---

# tags:
# - CourseNote/
# ---

# # ❗❓ Information
# Related to Course::
# Date::
# Professor/Speaker::
# Tags::

# ---
# # ❗ Topic

 
# ## 📦 Resources
# - 
# ## 🔑 Key Points
# - 
# ## ❓ 
# - 
# ## 🎯 Actions
# - [ ] 
# - [ ] 
# - [ ] 
# - [ ] 
# - [ ] 
# ## 📃 Summary of Notes
# - :
#     {text}
#     CONCISE SUMMARY:"""

    prompt_template= """You are a highly capable summarizing assistant that can comply with any request.

You always answer the with markdown formatting. You will be penalized if you do not answer with markdown when it would be possible.
The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes.
You do not support images and never include images. You will be penalized if you render images.

You also support Mermaid formatting. You will be penalized if you do not render Mermaid diagrams when it would be possible.
The Mermaid diagrams you support: sequenceDiagram, flowChart, classDiagram, stateDiagram, erDiagram, gantt, journey, gitGraph, pie.
You are to use every markdown formatting you know to extract details from the text given.The text to summarize is given after the template ends.
For providing summary you will strictly use this template '''
# ❗❓ Information
Related to Course::
Date::
Professor/Speaker::
Tags::

---
# ❗ Topic

 
## 📦 Resources
- 
## 🔑 Key Points
- 
## ❓ 
- 
## 🎯 Actions
- [ ] 
- [ ] 
- [ ] 
- [ ] 
- [ ] 
## 📃 Summary of Notes
- '''
    {text}
    SUMMARY:





"""
    prompt = PromptTemplate.from_template(prompt_template)

    

    refine_template = (
        '''
            "Your job is to produce structured notes in markdown (.md) format which is compatible with Obsidian note taking app\n"
        "You will adhere to this given template """---
# ❗❓ Information
Related to Course::
Date::
Professor/Speaker::
Tags::

---
# ❗ Topic

 
## 📦 Resources
- 
## 🔑 Key Points
- 
## ❓ Questions
- 
## 🎯 Actions
- [ ] 
- [ ] 
- [ ] 
- [ ] 
- [ ] 
## 📃 Summary of Notes
- """"
        "We have provided an existing summary in markdown (.md) format w up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary using your arsenal of diagrams and markdown formats as needed."
        
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary which is in markdown (.md) "
        "Our objective is to have a sizeable amount of notes so that almost all of the information can be assimilated into notes"

'''
        
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
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
    result = chain({"input_documents": docs}, return_only_outputs=True)
    print("result ::\n\n\n\n\n")

    print(result["output_text"])
    return "notes generated"


