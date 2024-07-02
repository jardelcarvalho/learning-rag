import os

import chainlit as cl
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig



def _load_api_token(file):
    with open(file, 'r') as f:
        key = f.read().replace(' ', '').replace('\n', '').replace('\t', '')
    return key



api_token = _load_api_token('./api_token.txt')



model_id = 'gpt2-medium'

#Loading a conversational model
conv_model = HuggingFaceEndpoint(huggingfacehub_api_token=api_token, 
                                 repo_id=model_id,
                                 temperature=.1,
                                 max_new_tokens=200)



# template = '''You are a helpful assistant that translates a English text to Portuguese.

# {query}
# '''
# prompt = PromptTemplate(template=template, input_variables=['query'])

prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system', 
            "You are a helpful assistant that translates a English text to Portuguese."
        ),
        ('human', '{text}')
    ]
)






@cl.on_chat_start
async def on_chat_start():
    conv_chain = prompt | conv_model | StrOutputParser()
    cl.user_session.set('llm_chain', conv_chain)


@cl.on_message
async def on_message(message):
    llm_chain = cl.user_session.get('llm_chain')

    msg = cl.Message(content="")

    async for chunk in llm_chain.astream(
        {'text': message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

#To run:
#chainlit run <filename.py> -w --port xxxx

    
