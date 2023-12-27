import logging
import uvicorn
import json
import constants
import asyncio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from agent import get_agent
from tools import search, ascrape_multiple_websites, allm_rank_chain

app = FastAPI()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)  

@app.get('/search-agent', description="use agent to search in the web")
def agent_search(query: str):
    agent = get_agent()
    content = agent.run({"input": "your research objective is objective:" + query})   
    return content

@app.get('/search-v1', description="simple search with llm description extraction")
async def simple_search(query: str):
    search_result = json.loads(search(query))
    gen = ascrape_multiple_websites(list(set(res['link'] for res in search_result['organic'][:constants.TOP_K])), query)
    return StreamingResponse(gen, media_type='text/event-stream')

@app.get('/search-v2', description="let llm figure out best query to search in google pick top relevent site and extract")
async def llm_search(query: str):
    # let llm figure out the query
    # search those query in google
    # let llm pick and rank top k result or consolidate result to remove duplicate
    # start scraping the web
    # stream the result one by one for better user performancez
    search_result = json.loads(await allm_rank_chain(query))
    gen = ascrape_multiple_websites(search_result, query, map_reduce=True)
    return StreamingResponse(gen, media_type='text/event-stream')
    
if __name__ == '__main__':
    uvicorn.run('main:app', reload=True)