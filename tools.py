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
from prompts import EXTRACT_PROMPT, MAP_PROMPT, REDUCE_PROMPT, SEARCH_QUERY_PROMPT, SEARCH_RESULT_RANK_PROMPT


@timeit
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': constants.SERP_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text


@timeit
def scrape(url, query):
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={constants.BROWSERLESS_API_KEY}"
    response = requests.post(post_url, headers=headers, data=data_json)
    
    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", api_key=constants.OPENAI_API_KEY)

    
        map_prompt = """
            Collect exact text content that is description of product "{query}".
            description: {text}
        """
        map_prompt_template = PromptTemplate(
            template=map_prompt, input_variables=["query", "text"])

        llm_chain = LLMChain(
            llm=llm,
            prompt=map_prompt_template,
            verbose=True,
            
        )

        output = llm_chain.run(text=text, query=query)
        return output
    else:
        print(f"HTTP request failed with status code {response.status_code}")
  
# v1
async def ascrape_multiple_websites(urls: List[str], query: str, map_reduce = False):
    """
        loop throught urls, call browser to get render html, parse using bs and extract with openAI
        all are done concurrently and stream the result back to the client as it is done.
    """  
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", api_key=constants.OPENAI_API_KEY)

    extract_prompt_template = PromptTemplate(
            template=EXTRACT_PROMPT, input_variables=["query", "text"])

    llm_chain = LLMChain(
            llm=llm,
            prompt=extract_prompt_template,
            verbose=False,   
        )
    
    loop = asyncio.get_event_loop()
    
    def request(url: str):
        logging.info(f'scraping {url}')
        post_url = f"https://chrome.browserless.io/content?token={constants.BROWSERLESS_API_KEY}"
        data = {
            "url": url,
        }
        headers = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
        }
        data_json = json.dumps(data)
        response = requests.post(post_url, headers=headers, data=data_json)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.decompose()
            text = soup.get_text()
            
            if len(text) > 20000 and map_reduce:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
                chunks = text_splitter.create_documents([text])
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type='map_reduce',
                    verbose=False,
                    map_prompt=PromptTemplate(
                        template=MAP_PROMPT, 
                        input_variables=['text', 'query']
                        ).partial(query=query),
                    combine_prompt=PromptTemplate(
                        template=REDUCE_PROMPT, 
                        input_variables=['text'])
                )
                # running this map_reduce concurrently to call gpt api would be nice to have
                output = chain.run(chunks)
                return {'description': output, "url": url, "reduced": True}
            else:
                output = llm_chain.run(text=text[:2000], query=query)
                return {'description': output, "url": url, "reduced": False}
        else:
            print(f"Status: {response.status_code}, content: {response.content}")
            
    for f in asyncio.as_completed([loop.run_in_executor(None, request, url) for url in urls]):
        result = await f
        logging.info('sending result for {url}'.format(url=result.get('url')))
        yield json.dumps(result)

#v2
async def multi_search(queries: List[str]):
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
    return await asyncio.gather(*[loop.run_in_executor(None, single_search, query) for query in queries])

   
async def llm_rank_chain(query: str):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", api_key=constants.OPENAI_API_KEY)
    query_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(template=SEARCH_QUERY_PROMPT, input_variables=['query']),
        verbose=True
    )
    queries = json.loads(query_chain.run(query=query))
    search_result = await multi_search(queries)
    rank_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=SEARCH_RESULT_RANK_PROMPT, 
            input_variables=['query', 'top_k', 'result']
        ).partial(query=query, top_k=constants.TOP_K),
        verbose=True
    )
    urls = rank_chain.run(result=search_result)
    
    return urls