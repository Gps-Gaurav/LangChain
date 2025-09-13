from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key="AIzaSyAlU06lSQ8CfPuh6ry0WnJjL6MdYTfRvoE"
)

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", 
                           [
        "ResNet: Deep Residual Learning for Image Recognition",
        "GANs: Generative Adversarial Nets",
        "Vision Transformers (ViT): An Image is Worth 16x16 Words",
        "AlphaGo: Mastering the Game of Go with Deep Neural Networks and Tree Search"
    ] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('prompts/template.json')



if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)