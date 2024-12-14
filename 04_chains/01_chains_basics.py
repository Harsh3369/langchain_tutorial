#load dependencies
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

#load env variables
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini",temperature= 0.7)

#Part 1: Create a chatprompttemplate using a template string

template = "Provide me interview tips for {profile} with {year_of_experience} years of experience."
prompt_template = ChatPromptTemplate.from_template(template)

#############Will trigger the below code using Chains ############################################
# print("-----Prompt from template-------")
# prompt = prompt_template.invoke({"profile" : "ML engineer", "year_of_experience": 7})
# print(prompt)

# output = model.invoke(prompt)
# print(output.content)

##################################################################################################

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"profile" : "ML engineer", "year_of_experience": 7})

print(result)