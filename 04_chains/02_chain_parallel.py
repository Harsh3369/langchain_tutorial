from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI

#LOAD ENV VARIABLES
load_dotenv()

#create a llm model
model = ChatOpenAI(model= 'gpt-4o-mini', temperature= 0.2)

#define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'You are an AI Engineering and ML Engineering Interview Expert.'),
    ('human', 'list some major features/topics a cadidate should prepare for an upcoming interview for {interview_profile} role.'),
])

#Define character to show
def characters_to_show(num_of_years):
    pos_char_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI Engineering and ML Engineering Interview Expert."),
            ("human", "Given these experience_category: {num_of_years} years, list some behaviour/characters the candiadate should pay attention too for interview."),
        ]
    )
    return pos_char_template.format_prompt(num_of_years = num_of_years)

#Define character not to show
def characters_not_to_show(num_of_years):
    neg_char_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI Engineering and ML Engineering Interview Expert."),
            ("human", "Given these experience_category: {num_of_years} years, list some behaviour/characters the candiadate should avaoid during the interview."),
        ]
    )
    return neg_char_template.format_prompt(num_of_years = num_of_years)

# Combine pros and cons into a final review
def combine_pos_neg(pos, neg):
    return f"positive character:\n{pos}\n\nNegative character:\n{neg}"

# Simplify branches with LCEL
pos_branch_chain = (
    RunnableLambda(lambda x: characters_to_show(x)) | model | StrOutputParser()
)

neg_branch_chain = (
    RunnableLambda(lambda x: characters_not_to_show(x)) | model | StrOutputParser()
)

#ccreate the combined chain using langchain
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"positive": pos_branch_chain, "negative": neg_branch_chain})
    | RunnableLambda(lambda x: combine_pos_neg(x["branches"]["positive"], x['branches']['negative']))
)

#run the chain
result = chain.invoke({"interview_profile": "ML Engineer", "num_of_years": 7})
print(result)