import os
from dotenv import load_dotenv
from rich import print
from langchain.chat_models import init_chat_model 

load_dotenv()

MODEL_NAME = "gpt-5-nano"
MODEL_PROVIDER = "openai"
llm = init_chat_model(model = MODEL_NAME, model_provider = MODEL_PROVIDER)
response = llm.invoke("Hi there, how are you?")
response_1 = llm.invoke("What is the capital of France?")

print(response.content)
print(response_1.content)

