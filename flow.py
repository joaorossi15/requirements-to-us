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

An Ethical User Story must follow this exact structure:

Title: <short title>

Description:
As a <persona>, I want <capability> so that <ethical benefit>.

Work:
<implementation task 1>
<implementation task 2>
<implementation task 3>

Rules:
- Return exactly one Title.
- Return exactly one Description.
- Return exactly 2 to 3 Work/Acceptance Criteria items.
- If more ideas exist for Work/Acceptance Criteria, select the 3 most important ones.
- Each Work item must be written on a single line.
- Do not use bullet points, numbering, or extra labels.
- Do not explain your answer.
- Do not repeat the requirement.
- The persona must be a realistic stakeholder, such as a user, developer, administrator, auditor, or regulator.
- The Description must clearly translate the ethical requirement into a user need.
- Each Work item must describe a concrete implementation task for developers.
- Work items must be specific, practical, and implementable.
- Avoid vague expressions such as "ensure transparency", "promote fairness", "support accountability", or "be ethical".
- Focus on concrete system behavior, user interface elements, logs, validations, stored data, notifications, review steps, or access controls.

Context:
{context}

Requirement:
{requirement}

Return the Ethical User Story exactly in this format:

Title: ...

Description:
As a ..., I want ... so that ...

Work:
...
...
...

Example:

Title: Human Review of High Impact AI Decisions

Description:
As a system administrator, I want automated decisions that significantly impact users to be reviewable so that harmful or incorrect outcomes can be prevented.

Work:
Store automated decisions and their input data in an audit log.
Provide an interface where administrators can review flagged decisions.
Allow administrators to override or cancel automated decisions before execution.

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
