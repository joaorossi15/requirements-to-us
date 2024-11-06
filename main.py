from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.vectorstores.chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from rag.rag import generate_store
import transformers

def load_model():
    mistral = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
    model = PeftModel.from_pretrained(mistral, "joaorossi15/mistral-7B-ai-ethics")
    return model

def model_rag(req: str, persist_path: str):
    model = load_model()
    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=transformers.AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"),
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=300,
    )

    prompt_template = """
    ### [INST] 
    Instruction: Transform the requirement below into a one line brief description of an ethical user story:

    {context}

    ### QUESTION:
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
    query = f'[INST] Transform the requirement below into a one line brief description of an ethical user story \\n{req} [/INST]'
    results = db.similarity_search_with_relevance_scores(query, k=3)

    rag_chain = ({"context": results, "requirement": RunnablePassthrough()} | llm_chain)
    rag_chain.invoke(req)

