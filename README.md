# RAG agent with Neo4j

This is a repo to showcase how to quickly setup a RAG agent with Neo4j graph database. It include vector search and graph RAG using Neo4j's Cypher queries.

I just followed a tutorial to create a RAG agent working with a neo4j DB. 

The added value of this repo is: 
- I updated the code (imports/API usage were outdated)
- They created an agent with the deprecated langchain's agent object, whereas I created the agent with Huggingface's smolagents.
- Its simplicity.

Reference: [Neo4j tutorial](https://neo4j.com/developer-blog/knowledge-graph-rag-application/)

## Steps
0. install everything for desktop
1. create db
2. run the connection script to ensure that it works
3. try the search script

## 1. Install neo4j desktop and create a db
1. download and install free neo4j desktop for windows
2. create a new dbms and create a DB with the test data
    - test data: https://gist.githubusercontent.com/tomasonjo/08dc8ba0e19d592c4c3cde40dd6abcc3/raw/da8882249af3e819a80debf3160ebbb3513ee962/microservices.json
3. select it and go to plugins on the right panel and install APOC extension

## 2. Create the .env file
Template: 
```
OPENAI_API_KEY=
NEO4J_DB_URL="bolt://localhost:PORT"
NEO4J_DB_USERNAME=
NEO4J_DB_PWD=
HUGGINGFACEHUB_API_TOKEN=
MISTRAL_API_KEY=
```

Run the _1_connect.py script to ensure connection to db works.

**About the models**:
- I use mistral AI API to perform inference (it can easily be embedded as a langchain inference object)
- I use HF API to run the agent
- I use sentence-transformers' "all-MiniLM-L6-v2" model to perform text embedding locally 

## 3. Create a new Python env

## 4. Try the different arguments with _2_vector_search.py

```bash
usage: _2_vector_search.py [-h] option

A script to interact with a Neo4j graph database using various methods like
vector search, QA chain, Cypher QA chain, and tool agent.

positional arguments:
  option      Choose one of the following options: ['vector', 'qa',
              'cypher_qa', 'code_agent']
```

```uv run _2_vector_search.py vector```

```uv run _2_vector_search.py qa```

```uv run _2_vector_search.py cypher_qa```

```uv run _2_vector_search.py code_agent```
