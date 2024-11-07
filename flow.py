from enum import auto
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.vectorstores.chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import transformers

def load_model():
    mistral = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    model = PeftModel.from_pretrained(mistral, "joaorossi15/mistral-7B-v03-ai-ethics-translator")
    return model

def model_rag(persist_path: str):
    model = load_model()
    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3"),
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=300,
        device=0
    )

   # prompt_template = """
    ### [INST] 
    #Instruction: Basing your answer on the ethical principles of: transparency, non-maleficence, responsibility, privacy, beneficence, freedom and autonomy, sustainability, dignity and justice, transform the requirement below into a one line brief description of an ethical user story:

    #{context}

    ### QUESTION:
    #{requirement} 

    #[/INST]
    #"""
    
    prompt_template = """
    ### [INST] 
    Instruction: Basing your answer on the AI principles related to the specific context defined below of the requirement specifie, transform this ethical requirement into a one line brief description of an ethical user story:

    {context}

    Now given the context, transform the requirement below into a one line brief description of an ethical user story.

    ### REQUIREMENT:
    {requirement} 

    [/INST]
    """

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    prompt = PromptTemplate(
        input_variables=["context", "requirement"],
        template=prompt_template,
    )

    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

    db = Chroma(persist_directory=persist_path, embedding_function=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

    rag_chain = ({"context": db.as_retriever(), "requirement": RunnablePassthrough()} | llm_chain)
            
    return rag_chain
