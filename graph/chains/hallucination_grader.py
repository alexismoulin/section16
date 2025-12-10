from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-5.1", temperature=0)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination presence in the generated answer."""
    binary_score: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'.")

structured_llm_grader = llm.with_structured_output(schema=GradeHallucinations)

system = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    messages=[
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
