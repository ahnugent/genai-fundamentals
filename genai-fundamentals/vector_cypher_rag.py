import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import VectorCypherRetriever

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# target = "actors"
target = "directors"

# Define retrieval query

if target == "actors":
    retrieval_query = """
    MATCH (node)<-[r:RATED]-()
    RETURN 
    node.title AS title, node.plot AS plot, score AS similarityScore, 
    collect { MATCH (node)-[:IN_GENRE]->(g) RETURN g.name } as genres, 
    collect { MATCH (node)<-[:ACTED_IN]->(a) RETURN a.name } as actors, 
    avg(r.rating) as userRating
    ORDER BY userRating DESC
    """
elif target == "directors":
    retrieval_query = """
    MATCH (node)<-[r:RATED]-()
    RETURN 
    node.title AS title, node.plot AS plot, score AS similarityScore, 
    collect { MATCH (node)-[:IN_GENRE]->(g) RETURN g.name } as genres, 
    collect { MATCH (node)<-[:DIRECTED]-(director) RETURN director.name } as directors, 
    avg(r.rating) as userRating
    ORDER BY userRating DESC
    """

# Create retriever
retriever = VectorCypherRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

#  Create the LLM
llm = OpenAILLM(model_name="gpt-4o")

# Create GraphRAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Search ...

if target == "actors":
    query_text = "Find the highest rated action movie about travelling to other planets"
    # query_text = "Find the lowest rated comedy movie about vampires"
    # query_text = "Find the lowest rated comedy movie" #: not reliable: top_k=5 reduces subset, letting similarity score bias results randomly; don't use LLM for this case! 
    # query_text = "What genres are represented about movies where the hero fails his mission?"
elif target == "directors":
    query_text = "Who has directed movies about weddings?"

response = rag.search(
    query_text=query_text, 
    retriever_config={"top_k": 5},
    return_context=True
)

print(response.answer)
print("CONTEXT:", response.retriever_result.items)

# Close the database connection
driver.close()