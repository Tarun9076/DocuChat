from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Set up Azure OpenAI model
import os
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not api_key or not endpoint or not deployment_name or deployment_name == "your_azure_openai_deployment_name_here":
    raise ValueError(
        "Azure OpenAI configuration is missing or incomplete in your .env file.\n"
        "Please ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME are correctly set."
    )

model = AzureChatOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
    azure_deployment=deployment_name,
)

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
    
    # Step 3: Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()