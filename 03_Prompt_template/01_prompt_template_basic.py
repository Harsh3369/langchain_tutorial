from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

#Part 1: Create a chatprompttemplate using a template string

template = "Provide me interview tips for {profile} with {year_of_experience} years of experience."
prompt_template = ChatPromptTemplate.from_template(template)

print("-----Prompt from template-------")
prompt = prompt_template.invoke({"profile" : "ML engineer", "year_of_experience": 7})
print(prompt)

#Part 2: Prompt template with system and Human Messages (using Tuples)

message = [
    ("system", "You are a Machine learning and AI Engineering Expert who has to take interview for candidate for {profile} role in your team."),
    ("human","Tell me {top_n} topics I sould prepare as a {experience_category} data scientist/ ML engineer."),
]

prompt_template_v2 = ChatPromptTemplate.from_messages(message)
prompt_v2 = prompt_template_v2.invoke({"profile" : "ML engineer", "top_n": 5, "experience_category": "senior"})

print("-------Prompt Template for Usecase 2----------")
print(prompt_v2)