from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import BaseRetriever
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import text

class JobSearchRetriever(BaseRetriever):
    def __init__(self, db: Session, embed_model):
        self.db = db
        self.embed_model = embed_model
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self.embed_model.embed(query_bundle.query_str)
        results = self.query_job_embeddings(self.db, query_embedding)
        
        nodes = []
        for result in results:
            node = TextNode(
                text=f"""
                Title: {result.title}
                Company: {result.company_name}
                Location: {result.processed_location}
                Salary: ${result.processed_min_salary:,.0f} - ${result.processed_max_salary:,.0f}
                Category: {result.job_category}
                Description: {result.processed_description[:200]}...
                """,
                id_=str(result.id)
            )
            nodes.append(NodeWithScore(node=node, score=result.similarity_score))
        
        return nodes
    @staticmethod
    def query_job_embeddings(db: Session, embedding: list[float], limit: int = 5):
        embedding_string = ','.join(map(str, embedding))
        sql = text(f"""
            SELECT *,
                1 - (description_embedding <=> ARRAY[{embedding_string}]::vector) AS similarity_score
            FROM job_postings
            WHERE processed_min_salary IS NOT NULL
            ORDER BY description_embedding <=> ARRAY[{embedding_string}]::vector
            LIMIT :limit
        """)
        return db.execute(sql, {"limit": limit}).fetchall()
