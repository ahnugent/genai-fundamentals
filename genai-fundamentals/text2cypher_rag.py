import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import Text2CypherRetriever

# Use examples in call to build Cypher query?
use_examples = True
use_schema = False
print(f"OPTION: use examples: {use_examples}")
print(f"OPTION: use schema: {use_schema}")

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create Cypher LLM 
t2c_llm = OpenAILLM(
    model_name="gpt-4o", 
    model_params={"temperature": 0}
)

if use_schema:
    # Specify your own Neo4j schema:
    neo4j_schema = """
    Node properties:
    Person {name: STRING, born: INTEGER}
    Movie {tagline: STRING, title: STRING, released: INTEGER}
    Genre {name: STRING}
    User {name: STRING}

    Relationship properties:
    ACTED_IN {role: STRING}
    RATED {rating: INTEGER}

    The relationships:
    (:Person)-[:ACTED_IN]->(:Movie)
    (:Person)-[:DIRECTED]->(:Movie)
    (:User)-[:RATED]->(:Movie)
    (:Movie)-[:IN_GENRE]->(:Genre)
    """

# Build the retriever ...
if use_examples:
    # Cypher examples as input/query pairs: 
    examples = [
        #E: "USER INPUT: 'How many movies are in the Sci-Fi genre?' QUERY: MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: 'Sci-Fi'}) RETURN 'There are ' || toString(count(m) AS numberOfMovies) || ' movies in the ' || g.name || ' genre.' AS result",
        #X: "USER INPUT: 'How many movies are in the Sci-Fi genre?' QUERY: MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: 'Sci-Fi'}) RETURN g.name, count(m) AS numberOfMovies",
        "USER INPUT: 'Get movies in the Sci-Fi genre?' QUERY: MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: 'Sci-Fi'}) RETURN g.name AS genre, count(m) AS numberOfMovies",
        "USER INPUT: 'Get user ratings for a movie?' QUERY: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE m.title = 'Movie Title' RETURN r.rating"
    ]
    t2cr_args = {'driver': driver, 'llm': t2c_llm, 'examples': examples, 'neo4j_schema': None}
else:
    t2cr_args = {'driver': driver, 'llm': t2c_llm, 'examples': None, 'neo4j_schema': None}

if use_schema:
    t2cr_args['neo4j_schema'] = neo4j_schema

# retriever = Text2CypherRetriever(
#     driver = t2cr_args['driver'],
#     llm = t2cr_args['llm'],
#     examples = t2cr_args['examples'],
#     neo4j_schema = t2cr_args['neo4j_schema'],
# )

# print(f'Examples: {examples}')  #: check to see if examples is damaged

retriever = Text2CypherRetriever(
    driver=driver,
    llm=t2c_llm,
    examples=examples,
)


llm = OpenAILLM(model_name="gpt-4o")
rag = GraphRAG(retriever=retriever, llm=llm)

# query_text = "Which movies did Hugo Weaving star in?"
# query_text = "Who directed the movie Superman?"
query_text = "How many movies are in the Sci-Fi genre?"
# query_text = "What are examples of Action movies?"

response = rag.search(
    query_text=query_text,
    return_context=True
    )

print(response.answer)
print("CYPHER :", response.retriever_result.metadata["cypher"])
print("CONTEXT:", response.retriever_result.items)

driver.close()

"""
FINDINGS
-----------
1. Example query must exactly match "USER INPUT" key.
2. Must include the contextual keyword in the example code (e.g. "genre") so that LLM understands the number that was returned.

FUTURE EXPERIMENTS
---------------------
1. Use few-shot or RAG to map query to examples, then append a query based on the example.

"""
