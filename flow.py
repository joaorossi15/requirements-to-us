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

You are an expert in software requirements engineering and AI ethics.

Your task is to transform an ethical requirement for an AI system into an **Ethical User Story (EUS)**.

Ethical User Stories translate ethical principles into actionable software requirements using the Agile user story format.

An Ethical User Story must contain two parts:

1. USER STORY
As a <persona or stakeholder>,
I want <capability>,
so that <ethical benefit or protection>.

2. ACCEPTANCE CRITERIA
Write 2–4 acceptance criteria describing what must be implemented for the requirement to be satisfied.
Use the Given / When / Then format whenever possible.

Guidelines:
- The persona should be a realistic stakeholder (user, developer, regulator, system operator, etc.).
- The capability should operationalize the ethical principle into a system feature.
- The benefit should clearly describe the ethical goal.
- Acceptance criteria should be concrete and implementable by developers.
- Avoid vague language such as "be ethical".
- Focus on practical system behavior.

Context (ethical principles and background):
{context}

Ethical requirement:
{requirement}

Generate the Ethical User Story following this structure exactly:

Title: <short title>

Description:
As a <persona>,
I want <capability>,
so that <benefit>.

Work (Acceptance Criteria):
- Given ...
- When ...
- Then ...

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
