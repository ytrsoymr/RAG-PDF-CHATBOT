import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config import GOOGLE_API_KEY


# Load Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

# Define a structured prompt template
PROMPT_TEMPLATE = """
You are an AI assistant that answers user queries based strictly on the given context.

CONTEXT INFORMATION:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer the question based ONLY on the context.
- Do NOT add any information beyond what is in the context.
- Do NOT include phrases like "according to the context" or "mentioned in the context".
- Provide a direct and precise response.
"""


prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def generate_answer(query: str, context: str):
    """Generate AI response based on retrieved context."""
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content
