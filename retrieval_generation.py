from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI
from langsmith import traceable
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ecommbot.ingest import ingestdata


load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')


def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    BOT_TEMPLATE = """
    You are a retrieval-augmented generation (RAG) assistant.

    Use only the retrieved context to answer the question.
    If unsure, output "Not enough information."

    Output strictly valid JSON with this format:
    {
    "answer": "<string>",
    "contexts": ["<string>", "<string>", ...],
    "reasoning": "<string>"
    }

    CONTEXTS:
    {context}

    QUESTION:
    {question}
    """


    prompt = ChatPromptTemplate.from_template(BOT_TEMPLATE)

    llm = ChatGroq(groq_api_key=groq_api_key,
             model_name="llama-3.3-70b-versatile",
             temperature=0.5)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__=='__main__':
    vstore = ingestdata("done")
    chain  = generation(vstore)
    print(chain.invoke("can you tell me the best bluetooth buds?"))
    
    
    
    