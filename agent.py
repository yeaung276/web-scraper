from typing import Type
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory

import constants
from prompts import agent_prompts
from tools import search, scrape


def get_agent():
    class ScrapeWebsiteInput(BaseModel):
        """Inputs for scrape_website"""
        product_name: str = Field(
            description="The name of the product searching")
        url: str = Field(description="The url of the website to be scraped")
        
    class WebSearchInput(BaseModel):
        "Inputs for web_search"
        product_name: str = Field(description="The name of the product to search")


    class ScrapeWebsiteTool(BaseTool):
        name = "scrape_website"
        description = "useful when you need to get scrape exact product description, passing both url and product name to the function; DO NOT make up any url, the url should only be from the search results"
        args_schema: Type[BaseModel] = ScrapeWebsiteInput

        def _run(self, product_name: str, url: str):
            return scrape(url, product_name)

        def _arun(self, url: str):
            raise NotImplementedError("error here")
        
    class WebSearchTool(BaseTool):
        name = "web_search"
        description = "useful for when you need to search for informations"
        args_schema: Type[BaseModel] = WebSearchInput
        
        def _run(self, product_name: str):
            return search(product_name)
        
        def _arun(self, product_name: str):
            raise NotImplementedError()


    # 3. Create langchain agent with the tools above
    tools = [
        WebSearchTool(),
        ScrapeWebsiteTool(),
    ]

    system_message = SystemMessage(
        content=agent_prompts['v2']
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_message,
    }

    llm = ChatOpenAI(
        temperature=0, 
        model="gpt-3.5-turbo-16k-0613", 
        api_key=constants.OPENAI_API_KEY)
    memory = ConversationSummaryBufferMemory(
        memory_key="memory", return_messages=True, llm=llm, max_token_limit=8000)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        max_iterations=3,
        early_stopping_method="generate",
        memory=memory,
    )
    return agent

