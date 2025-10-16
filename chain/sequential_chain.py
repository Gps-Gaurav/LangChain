from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 
from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1=PromptTemplate(
    template="What is a good name for a company that makes {product}?",
    input_variables=["product"]
)
prompt2=PromptTemplate(
    template="What is a good slogan for a company named {company_name}?",
    input_variables=["company_name"]
)       

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key="AIzaSyAlU06lSQ8CfPuh6ry0WnJjL6MdYTfRvoE"
)

parser= StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"product": "laptop stickers"})
print(result)

chain.get_graph().print_ascii()