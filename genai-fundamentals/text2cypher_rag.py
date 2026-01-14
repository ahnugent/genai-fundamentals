"""
Docstring for genai-fundamentals.text2cypher_rag
"""
# 
"""
NEW FEATURES
---------------
1. Refactored retriever composition: arguments are assigned via a dict to allow any combination of optional args.

FINDINGS
-----------
1. Example query must exactly match "USER INPUT" key in order to be used by retriever.
2. Must include the contextual keyword in the example code (e.g. "genre") so that LLM understands the number that was returned.

FUTURE EXPERIMENTS
---------------------
1. Use few-shot or RAG to map query to examples, then append a query based on the example.
2. Get output to indicate whether an example was used.
3. How to get output to indicate quality ot match between query and examples?

"""

import os
import re
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import Text2CypherRetriever


def parse_input_and_query(examples_):
    user_input_match = re.search(r"USER INPUT:\s*'(.*?)'", examples_)
    query_match = re.search(r"QUERY:\s*(.*)", examples_)
    user_input = user_input_match.group(1) if user_input_match else None
    query = query_match.group(1).strip() if query_match else None
    return user_input, query

def get_match(query_, examples_, show = False):
    # Indicates whether the query is found in the examples.
    user_inputs = []
    for ex_ in examples_:
        user_inputs.append(parse_input_and_query(ex_)[0])
    matched_ = (query_ in user_inputs)
    if show:
        print(user_inputs)
        print(f'Match found: {matched_}')
    return matched_


# Use examples in call to build Cypher query?
use_examples = True
use_schema = True
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
        #X: "USER INPUT: 'Get movies in the Sci-Fi genre?' QUERY: MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: 'Sci-Fi'}) RETURN g.name AS genre, count(m) AS numberOfMovies",
        #E: "USER INPUT: 'How many movies are in the Sci-Fi genre?' QUERY: MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: 'Sci-Fi'}) RETURN 'There are ' || toString(count(m) AS numberOfMovies) || ' movies in the ' || g.name || ' genre.' AS result",
        "USER INPUT: 'How many movies are in the Sci-Fi genre?' QUERY: MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: 'Sci-Fi'}) RETURN g.name AS genre, count(m) AS numberOfMovies",
        "USER INPUT: 'Get user ratings for a movie?' QUERY: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE m.title = 'Movie Title' RETURN r.rating"
    ]
    t2cr_args = {'driver': driver, 'llm': t2c_llm, 'examples': examples, 'neo4j_schema': None}
else:
    t2cr_args = {'driver': driver, 'llm': t2c_llm, 'examples': None, 'neo4j_schema': None}

if use_schema:
    t2cr_args['neo4j_schema'] = neo4j_schema

retriever = Text2CypherRetriever(
    driver = t2cr_args['driver'],
    llm = t2cr_args['llm'],
    examples = t2cr_args['examples'],
    neo4j_schema = t2cr_args['neo4j_schema'],
)

llm = OpenAILLM(model_name="gpt-4o")
rag = GraphRAG(retriever=retriever, llm=llm)

# query_text = "Which movies did Hugo Weaving star in?"
# query_text = "Who directed the movie Superman?"
query_text = "How many movies are in the Sci-Fi genre?"
# query_text = "What are examples of Action movies?"

# indicate whether an example was used:
is_matched = get_match(query_text, examples) #: this is an inference
print(f'Query was matched (exactly) to examples: {is_matched}')

response = rag.search(
    query_text=query_text,
    return_context=True
    )

print(response.answer)
print("CYPHER :", response.retriever_result.metadata["cypher"])
print("CONTEXT:", response.retriever_result.items)

driver.close()
