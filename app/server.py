import os
import pickle
import sys

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the main directory to the Python path
sys.path.append(os.path.dirname(current_dir))

# Add the utils package to the Python path
utils_dir = os.path.join(os.path.dirname(current_dir), 'utils')
sys.path.append(utils_dir)

from utils.document_loader import DocumentLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from typing import List, Any, Union, Dict
from utils.vector_store import create_vector_store, get_local_store
from utils.grader import GraderUtils
from utils.graph import GraphState
from utils.generate_chain import create_generate_chain
from utils.nodes import GraphNodes
from utils.edges import EdgeGraph
from langgraph.graph import END, StateGraph
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# with open("documents_50.pkl", "rb") as file:
#     saved_docs = pickle.load(file)

# store = create_vector_store(saved_docs)
store = get_local_store('pytorch_vectorstore')
retriever = store.as_retriever()

llm = ChatOllama(model='llama3.1')

grader = GraderUtils(llm)
retrieval_grader = grader.create_retrieval_grader()
hallucination_grader = grader.create_hallucination_grader()
code_evaluator = grader.create_code_evaluator()
question_rewriter = grader.create_question_rewriter()

# Initiating the graph 
workflow = StateGraph(GraphState)
# Create an instatnce of the GraphNodes class 
graph_nodes = GraphNodes(llm, retriever, retrieval_grader, hallucination_grader, code_evaluator, question_rewriter)
# Create an instance of the EdgeGraph class 
edge_graph = EdgeGraph(hallucination_grader, code_evaluator)

# Define the nodes
workflow.add_node("retrieve", graph_nodes.retrieve)
workflow.add_node("grade_documents", graph_nodes.grade_documents)
workflow.add_node("generate", graph_nodes.generate)
workflow.add_node("transform_query", graph_nodes.transform_query)

# Build graph 
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    'grade_documents',
    edge_graph.decide_to_generate,
    {
        'transform_query': "transform_query",
        "generate": "generate"
    }
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    edge_graph.grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query"
    }
)
chain = workflow.compile()

# Create the FastAPI app
app = FastAPI(
    title="Torch server",
    version='1.0',
    description="An API server to answer questions regarding the Pytorch Developer Docs"
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: dict

add_routes(
    app,
    chain.with_types(input_type=Input, output_type=Output),
    path="/torch_chat"
)

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="localhost", port=8000)