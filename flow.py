from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def model_rag(persist_path: str):
    model = load_model()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    text_generation_pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.0,
        do_sample=False,
        repetition_penalty=1.1,
        max_new_tokens=300,
        return_full_text=False,
        device=0
    )

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    prompt_template = """
### [INST]

You are an expert in requirements engineering and AI ethics.

Your task is to transform an ethical AI requirement into an Ethical User Story (EUS).

Follow this exact template.

Title: <short title>

Description:
As a <persona>, I want <capability> so that <ethical benefit>.

Work (Acceptance Criteria):

Formatting rules:
- Write EXACTLY 3 acceptance criteria.
- Each acceptance criterion MUST be on a single line.
- Each line MUST follow the format:
  Given <system state> When <event> Then <system behavior>
- Do NOT break lines inside a criterion.
- Do NOT use bullet points.
- Do NOT explain anything.
- Acceptance criteria must describe concrete system behavior.

Context:
{context}

Requirement:
{requirement}

Return the Ethical User Story exactly in the specified format.

Title: Human Review of High Impact Decisions

Description:
As a system administrator, I want automated decisions that significantly impact users to be reviewable so that incorrect outcomes can be prevented.

Work (Acceptance Criteria):
Given the AI system generates a decision affecting a user When the decision confidence is below the configured threshold Then the system must flag the decision for human review.
Given a flagged decision exists When the administrator opens the review interface Then the system must display the input data, prediction, and confidence score.
Given the administrator rejects the automated decision When the rejection is confirmed Then the system must cancel the decision and log the override.

[/INST]
"""

    prompt = PromptTemplate.from_template(prompt_template)

    db = Chroma(
        persist_directory=persist_path,
        embedding_function=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    )

    retriever = db.as_retriever(search_kwargs={"k": 2})

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "requirement": RunnablePassthrough()
        }
        | prompt
        | mistral_llm
    )

    return rag_chain
