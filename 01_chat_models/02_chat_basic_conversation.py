from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini', temperature= 0)

message = [
    SystemMessage(content='You are a python developer assistant!'),
    HumanMessage(content="What is logging in python and why it is important?"),
]

response = model.invoke(message)
print(response.content)