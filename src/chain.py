from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate


# Custom prompt to keep answers grounded in documents
QA_PROMPT_TEMPLATE = """You are DocMind, an intelligent AI assistant that answers questions 
based strictly on the provided document context. 

Rules:
- Answer only from the given context
- If the answer is not in the context, say "I couldn't find this information in the uploaded documents."
- Be concise but complete
- Cite which part of the document supports your answer when possible
- Format your response clearly with bullet points or numbered lists when appropriate

Context from documents:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "chat_history", "question"]
)


def build_chain(vectorstore, groq_api_key: str):
    """
    Build a ConversationalRetrievalChain with:
    - Groq LLM (llama3-70b-8192 — fast and free)
    - FAISS retriever
    - Conversation memory (last 5 exchanges)
    """
    # LLM — Groq is fast and has generous free tier
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.2,        # Low temp = more factual answers
        max_tokens=1024
    )

    # Retriever — fetch top 4 most relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Memory — remembers last 5 Q&A pairs for context
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5
    )

    # Full RAG chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        verbose=False
    )

    return chain
