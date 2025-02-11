import os
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
from neo4j.exceptions import ServiceUnavailable, AuthError


load_dotenv()


def connect_to_neo4j():
    try:
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_DB_URL"), 
            username=os.getenv("NEO4J_DB_USERNAME"), 
            password=os.getenv("NEO4J_DB_PWD")
        )
        return graph
    except ServiceUnavailable as e:
        print(f"ServiceUnavailable error: {e}")
    except AuthError as e:
        print(f"Authentication error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def main():
    graph = connect_to_neo4j()
    if graph:
        print("Successfully connected to Neo4j")
    else:
        print("Failed to connect to Neo4j")


if __name__ == "__main__":
    main()
