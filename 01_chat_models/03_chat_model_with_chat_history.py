from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

#load env variables
load_dotenv()

#initiate the model from openai
model = ChatOpenAI(model='gpt-4o-mini', temperature= 0.2)

#Add empty list for chat history
chat_history = []

system_message = SystemMessage(content= 'You are a helpful AI assistant')
chat_history.append(system_message)

#enabling chat history
while True:
    query = input("You: ")
    if query.lower() == 'exit':
        break
    chat_history.append(HumanMessage(content=query))

    #getting response from llm
    response = model.invoke(chat_history)
    output = response.content
    chat_history.append(AIMessage(content=output))

    print(f"AI : {output}")

print("----MessageHistory------")
print(chat_history)
