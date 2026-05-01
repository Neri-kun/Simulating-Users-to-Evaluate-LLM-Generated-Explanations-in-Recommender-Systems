import logging
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_ollama import OllamaLLM

# Configure logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OllamaLLM (updated class)
ollama_model = OllamaLLM(model="gemma:2b", host="http://localhost:11434")

template = (
    "User ID: {userId}\n\n"
    "Average genre ratings:\n"
    "- Highly rated (≥ 4.0): {high_ratings}\n"
    "- Poorly rated (< 3.0): {low_ratings}\n\n"
    "Please predict how this user would rate the following 5 movies:\n"
    "{movies_to_rate}\n\n"
    "Use only the following rating values (in 0.5-star increments):\n"
    "0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0\n\n"
    "Respond with your predictions in this exact format (one per line):\n"
    "- Movie Title (Year): Predicted Rating\n\n"
    "If some information is missing, still provide your best estimate based on genre preferences or general trends."
)


recommendation_prompt = PromptTemplate(
    input_variables=["userId", "rating_history", "movies_to_rate", "high_ratings", "low_ratings"],
    template=template
)

# Use RunnableSequence instead of LLMChain
recommendation_chain = RunnableSequence(recommendation_prompt, ollama_model)