import os
from langchain_neo4j import Neo4jVector
# from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import GraphCypherQAChain
from sentence_transformers import SentenceTransformer
from langchain_neo4j import Neo4jGraph
from langchain.chains import RetrievalQA
from langchain_mistralai import ChatMistralAI
from smolagents import CodeAgent, HfApiModel, Tool
# from langchain_community.chat_models import ChatOpenAI
import time
import argparse

from _1_connect import connect_to_neo4j


OPTIONS = ['vector', 'qa', 'cypher_qa', 'code_agent']

parser = argparse.ArgumentParser(description="A script to interact with a Neo4j graph database using various methods like vector search, QA chain, Cypher QA chain, and tool agent.")
parser.add_argument(
    "option", type=str, help=f"Choose one of the following options: {OPTIONS}"
)
args = parser.parse_args()


load_dotenv()

# Free embedding model instead of openaiembeddings
print("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
class LocalEmbeddings:
    """Custom embedding wrapper to use SentenceTransformers like OpenAIEmbeddings."""
    def embed_documents(self, texts):
        return model.encode(texts, convert_to_list=True)  # Returns a list of vectors

    def embed_query(self, text):
        return model.encode([text], convert_to_list=True)[0]  # Single query vector
print("Loaded SentenceTransformer model.")

# Free llm model instead of chatopenai
models = [
    "ministral-8b-latest",  
    "ministral-3b-latest", 
    "mistral-small-latest",
]
hf_llm = ChatMistralAI(
    model=models[0],
    temperature=0,
    max_retries=2,
    max_tokens=512
)


def compute_vector_index():
    """
    - index_name: name of the vector index.
    - node_label: node label of relevant nodes.
    - text_node_properties: properties to be used to calculate embeddings and retrieve from the vector index.
    - embedding_node_property: which property to store the embedding values to.
    """

    print('Loading vector index...')
    vector_index = Neo4jVector.from_existing_graph(
        LocalEmbeddings(),  # OpenAIEmbeddings(),
        url=os.getenv("NEO4J_DB_URL"), 
        username=os.getenv("NEO4J_DB_USERNAME"), 
        password=os.getenv("NEO4J_DB_PWD"),
        index_name='tasks',
        node_label="Task",
        text_node_properties=['name', 'description', 'status'],
        embedding_node_property='embedding',
    )
    print('Loaded vector index.')
    return vector_index


def use_vector_index(vector_index):
    print('Using vector index')
    response = vector_index.similarity_search(
        "How will RecommendationService be updated?"
    )
    print(response[0].page_content)


def use_qa_chain(vector_index):
    print('Using QA chain')
    vector_qa = RetrievalQA.from_chain_type(
        llm=hf_llm,  # ChatOpenAI(),
        chain_type="stuff",
        retriever=vector_index.as_retriever()
    )
    output = vector_qa.invoke(
        "How will recommendation service be updated?"
    )
    print(output)
    # The RecommendationService is currently being updated to include a new feature 
    # that will provide more personalized and accurate product recommendations to 
    # users. This update involves leveraging user behavior and preference data to 
    # enhance the recommendation algorithm. The status of this update is currently
    # in progress.

    time.sleep(1)
    output = vector_qa.run(
        "How many open tickets are there?"
    )
    print(output)
    # There are 4 open tickets.


def use_cypher_qa_chain(graph):
    cypher_chain = GraphCypherQAChain.from_llm(
        cypher_llm = hf_llm,
        qa_llm = hf_llm, 
        graph=graph, 
        verbose=True,
        allow_dangerous_requests=True
    )
    questions = [
        "How many open tickets there are?",
        "Which team has the most open tasks?",
        "Which services depend on Database directly?"
    ]
    output = cypher_chain.invoke(
        questions[2]
    )
    print(output)



class TasksTool(Tool):
    name="TasksTool"
    description="""Useful when you need to answer questions about descriptions of tasks.
            Not useful for counting the number of tasks.
            Use full question as input.
            """
    inputs = {
        "prompt": {
            "type": "string",
            "description": "question about tasks",
        }
    }
    output_type = "string"

    def forward(self, prompt):
        vector_index = Neo4jVector.from_existing_graph(
            LocalEmbeddings(),
            url=os.getenv("NEO4J_DB_URL"), 
            username=os.getenv("NEO4J_DB_USERNAME"), 
            password=os.getenv("NEO4J_DB_PWD"),
            index_name='tasks',
            node_label="Task",
            text_node_properties=['name', 'description', 'status'],
            embedding_node_property='embedding',
        )
        vector_qa = RetrievalQA.from_chain_type(
            llm=hf_llm, 
            chain_type="stuff",
            retriever=vector_index.as_retriever()
        )
        return vector_qa.invoke(prompt)
    

class GraphTool(Tool):
    name="GraphTool"
    description="""Useful when you need to answer questions about microservices,
        their dependencies or assigned people. Also useful for any sort of 
        aggregation like counting the number of tasks, etc.
        Use full question as input.
        """
    inputs = {
        "prompt": {
            "type": "string",
            "description": "question about the graph content directly",
        }
    }
    output_type = "string"

    def forward(self, prompt):
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_DB_URL"), 
            username=os.getenv("NEO4J_DB_USERNAME"), 
            password=os.getenv("NEO4J_DB_PWD")
        )
        cypher_chain = GraphCypherQAChain.from_llm(
            cypher_llm = hf_llm,
            qa_llm = hf_llm, 
            graph=graph, 
            verbose=True,
            allow_dangerous_requests=True
        )
        return cypher_chain.invoke(prompt)


def agent():
    questions = [
        "Which team is assigned to maintain PaymentService?",
        "Which tasks have optimization in their description?"
    ]
    model = HfApiModel(model_id="meta-llama/Llama-3.3-70B-Instruct", token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))
    agent = CodeAgent(tools=[TasksTool(), GraphTool()], model=model, add_base_tools=False)
    response = agent.run(questions[1])
    print(response)


def main(option: str):

    assert option in OPTIONS

    match option:
        case 'vector':
            index = compute_vector_index()
            use_vector_index(index)
        case 'qa':
            index = compute_vector_index()
            use_qa_chain(index)
        case 'cypher_qa':
            graph = connect_to_neo4j()
            use_cypher_qa_chain(graph)
        case 'code_agent':
            agent()


if __name__ == "__main__":
    main(option=args.option)
