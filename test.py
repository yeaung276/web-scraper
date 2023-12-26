import time
import json
import logging
import asyncio
import requests
from typing import List
from bs4 import BeautifulSoup

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

import constants
from utils import timeit
from agent import get_agent
from tools import llm_rank_chain
from prompts import MAP_PROMPT, REDUCE_PROMPT, SEARCH_QUERY_PROMPT, SEARCH_RESULT_RANK_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log", mode='w'),
        logging.StreamHandler()
    ]
)  


# scraping
@timeit
def request(url: str, query: str):
    start = time.time()
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", api_key=constants.OPENAI_API_KEY)

    map_prompt = """
        Collect all sentences that is description of product "{query}". The sentences must not be altered in anyway.
        Here is the text:

        "{text}"
        DESCRIPTION:
        """
    map_prompt_template = PromptTemplate(
            template=map_prompt, input_variables=["query", "text"])

    llm_chain = LLMChain(
            llm=llm,
            prompt=map_prompt_template,
            verbose=True,   
        )
    post_url = f"https://chrome.browserless.io/content?token={constants.BROWSERLESS_API_KEY}"
    data = {
        "url": url
    }
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    data_json = json.dumps(data)
    response = requests.post(post_url, headers=headers, data=data_json)
    logging.info(f'browserless call: {time.time() - start} ms')
    start = time.time()
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style", "footer", "header", "nav"]):
            script.decompose()
        text = soup.get_text()
        logging.info(f'scraping: {time.time() - start} ms')
        start = time.time()
        if len(text) > 20000:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
            chunks = text_splitter.create_documents([text])
            chain = load_summarize_chain(
                llm=llm,
                chain_type='map_reduce',
                verbose=False,
                map_prompt=map_prompt_template.partial(query=query),
                combine_prompt=PromptTemplate(template=REDUCE_PROMPT, input_variables=['text'])
            )
            # running this map_reduce concurrently to call gpt api would be nice to have
            output = chain.run(chunks)
            logging.info(f'llm summary: {time.time() - start}')
            return {'description': output, "url": url, "reduced": True}
        else:
            output = llm_chain.run(text=text, query=query)
            logging.info(f'llm summary: {time.time() - start}')
            return {'description': output, "url": url, "reduced": False}
    else:
        print(f"HTTP request failed with status code {response.status_code}")

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", api_key=constants.OPENAI_API_KEY)

    map_prompt = """
        Collect all sentences that is description of product "{query}". The sentences must not be altered in anyway.
        Here is the text:

        "{text}"
        DESCRIPTION:
        """
    map_prompt_template = PromptTemplate(
            template=map_prompt, input_variables=["query", "text"])

    llm_chain = LLMChain(
            llm=llm,
            prompt=map_prompt_template,
            verbose=True,   
        )
    post_url = f"https://chrome.browserless.io/scrape?token={constants.BROWSERLESS_API_KEY}"
    data = {
        "url": url,
        "elements": [
            {
                "   "
            }
        ]
    }
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    data_json = json.dumps(data)
    response = requests.post(post_url, headers=headers, data=data_json)
    if response.status_code == 200:
        text = response.content
        logging.info(response.content)
        # if len(text) > 20000:
        #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
        #     chunks = text_splitter.create_documents([text])
        #     chain = load_summarize_chain(
        #         llm=llm,
        #         chain_type='map_reduce',
        #         verbose=False,
        #         map_prompt=map_prompt_template.partial(query=query),
        #         combine_prompt=PromptTemplate(template=REDUCE_PROMPT, input_variables=['text'])
        #     )
        #     # running this map_reduce concurrently to call gpt api would be nice to have
        #     output = chain.run(chunks)
        #     return {'description': output, "url": url}
        # else:
        #     output = llm_chain.run(text=text, query=query)
        #     return {'description': output, "url": url}
    else:
        print(f"HTTP status: {response.status_code}, content: {response.content}")      

# print(request('https://arizer.com/solo2/', 'Arizer Solo 2'))

# llm ranking
async def llm_rank_chain(query: str):
    start = time.time()
    def single_search(query: str):
        url = "https://google.serper.dev/search"

        payload = json.dumps({
            "q": query
        })

        headers = {
            'X-API-KEY': constants.SERP_API_KEY,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        return response.json()['organic'][:constants.TOP_K]
    loop = asyncio.get_event_loop()
    
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", api_key=constants.OPENAI_API_KEY)
    query_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(template=SEARCH_QUERY_PROMPT, input_variables=['query']),
        verbose=False
    )
    queries = json.loads(query_chain.run(query=query))
    logging.info(f'llm generate queries: {time.time() - start}ms')
    start = time.time()
    search_result = await asyncio.gather(*[loop.run_in_executor(None, single_search, query) for query in queries])
    logging.info(f'search result for each query: {time.time() - start} ms')
    start = time.time()
    rank_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=SEARCH_RESULT_RANK_PROMPT, 
            input_variables=['query', 'top_k', 'result']
        ).partial(query=query, top_k=constants.TOP_K),
        verbose=False
    )
    urls = rank_chain.run(result=search_result)
    logging.info(f'rank and pick top k by llm: {time.time() - start} ms')
    return urls

# print('result', asyncio.run(llm_rank_chain('Arizer Solo 2')))

# agent
def agent_search(query: str):
    agent = get_agent()
    return agent.run({"input": f"your research product is: {query}"})  

print('result', agent_search("Arizer Solo 2"))
 