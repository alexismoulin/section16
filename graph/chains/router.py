from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class RouteQuery(BaseModel):
    """Route a query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        default=...,
        description="Given a user question, choose to route it to websearch or vectorstore"
    )


llm = ChatOpenAI(model="gpt-5.1", temperature=0)
structured_llm_router = llm.with_structured_output(schema=RouteQuery)

system = """
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search.
"""

route_prompt = ChatPromptTemplate.from_messages(
    messages=[("system", system), ("human", "{question}")]
)

question_router = route_prompt | structured_llm_router