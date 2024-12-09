#using openai

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

response = model.invoke("What is langchain, in simple words?")

print(response.content)

##-------------------------------------------------------------##

#Using Huggingface

# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load the model and tokenizer from Hugging Face
# model_name = "bigscience/bloom-560m"  # You can choose other models if needed
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# prompt = 'What is langchain, in simple words?'

# # Tokenize the input
# inputs = tokenizer(prompt, return_tensors="pt")

# #Generate Output
# outputs = model.generate(inputs["input_ids"], 
#     max_length=50,  # Adjust the length of the response
#     num_return_sequences=1, 
#     temperature=0.2  # Adjust for creativity
# )

# # Decode and print the response
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(response)