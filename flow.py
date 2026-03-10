from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import transformers


def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3"
    )
    model = PeftModel.from_pretrained(
        base_model,
        "joaorossi15/mistral-7B-v03-ethical-us"
    )
    return model


def model_rag(persist_path: str):
    model = load_model()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    text_generation_pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.1,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=150,
        device=0
    )

    prompt_template = """
### [INST]
Instruction: Based on the AI principles related to the context below, transform the ethical requirement into an ethical user story following the TEMPLATE:

Title: <title>
Description: As a <persona> I want to <do something> so that <benefit>
Work: <acceptance criteria>

Context:
{context}

Requirement:
{requirement}

Generate the ethical user story following the TEMPLATE above.
[/INST]
"""

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    prompt = PromptTemplate.from_template(prompt_template)

    db = Chroma(
        persist_directory=persist_path,
        embedding_function=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    )

    retriever = db.as_retriever()

    rag_chain = (
        {
            "context": retriever,
            "requirement": RunnablePassthrough()
        }
        | prompt
        | mistral_llm
    )

    return rag_chain
