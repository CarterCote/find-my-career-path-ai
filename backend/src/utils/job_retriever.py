from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import BaseRetriever
from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text
from enum import Enum
from .job_graph import JobGraph
from ..models import JobPosting

class SearchStrategy(Enum):
    SEMANTIC = "semantic"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GRAPH = "graph"

class JobSearchRetriever(BaseRetriever):
    def __init__(self, db: Session, embed_model, strategy: SearchStrategy = SearchStrategy.SEMANTIC):
        self.db = db
        self.embed_model = embed_model
        self.strategy = strategy
        self.job_graph = JobGraph()
        super().__init__()
    
    def initialize_graph(self, jobs_df):
        """Initialize the job graph with data"""
        self.graph = self.job_graph.build_graph(jobs_df)
    
    def search_jobs(self, query: str, filters: Dict = None):
        """Main search function that uses the selected strategy"""
        if self.strategy == SearchStrategy.GRAPH:
            return self.graph_enhanced_search(query, filters)
        elif self.strategy == SearchStrategy.SEMANTIC:
            return self.semantic_search(query, filters)
        elif self.strategy == SearchStrategy.SPARSE:
            return self.sparse_search(query, filters)
        elif self.strategy == SearchStrategy.HYBRID:
            return self.hybrid_search(query, filters)
    
    def semantic_search(self, query: str, filters: Dict = None):
        """Original semantic search implementation"""
        query_embedding = self.embed_model.embed(query)
        
        sql = """
        WITH ranked_jobs AS (
            SELECT 
                *,
                1 - (description_embedding <=> :query_embedding::vector) AS semantic_score,
                ts_rank(to_tsvector('english', description), plainto_tsquery('english', :query)) AS text_score
            FROM postings
            WHERE 1=1
        """
        
        params = {"query": query, "query_embedding": query_embedding}
        
        if filters:
            if filters.get('min_salary'):
                sql += " AND min_salary >= :min_salary"
                params['min_salary'] = filters['min_salary']
            if filters.get('location'):
                sql += " AND location ILIKE :location"
                params['location'] = f"%{filters['location']}%"
        
        sql += """
            )
            SELECT * FROM ranked_jobs
            ORDER BY (semantic_score + text_score) DESC
            LIMIT :limit
        """
        params['limit'] = filters.get('limit', 10)
        
        return self.db.execute(text(sql), params).fetchall()
    
    def sparse_search(self, query: str, filters: Dict = None):
        """Quick text-based search"""
        sql = """
        SELECT 
            *,
            ts_rank(to_tsvector('english', title || ' ' || description), 
                   plainto_tsquery('english', :query)) AS relevance
        FROM postings
        WHERE 1=1
        """
        
        params = {"query": query}
        
        if filters:
            if filters.get('min_salary'):
                sql += " AND min_salary >= :min_salary"
                params['min_salary'] = filters['min_salary']
            if filters.get('location'):
                sql += " AND location ILIKE :location"
                params['location'] = f"%{filters['location']}%"
        
        sql += " ORDER BY relevance DESC LIMIT :limit"
        params['limit'] = filters.get('limit', 10)
        
        return self.db.execute(text(sql), params).fetchall()
    
    def hybrid_search(self, query: str, filters: Dict = None):
        """Combines sparse and semantic search"""
        # First get candidates from sparse search
        sparse_results = self.sparse_search(query, {"limit": 100, **filters})
        
        # Then rank with semantic search
        if not sparse_results:
            return []
            
        job_ids = [r.job_id for r in sparse_results]
        return self.semantic_search(query, {"job_ids": job_ids, **filters})
    
    def graph_enhanced_search(self, query: str, filters: Dict = None):
        """Enhance search results with graph-based recommendations"""
        # First get base results from semantic/sparse search
        base_results = self.semantic_search(query, filters)
        
        # Get related jobs through graph traversal
        enhanced_results = []
        seen_jobs = set()
        
        for job in base_results:
            job_id = job.job_id
            if job_id in seen_jobs:
                continue
                
            # Add the base job
            enhanced_results.append(job)
            seen_jobs.add(job_id)
            
            # Find related jobs through title and company connections
            if hasattr(self, 'graph'):
                related_jobs = self.job_graph.get_related_jobs(job_id, max_distance=2)
                for related_job in related_jobs:
                    if related_job['job_id'] not in seen_jobs:
                        # Get full job details from database
                        job_details = self.db.query(JobPosting).filter(
                            JobPosting.job_id == related_job['job_id']
                        ).first()
                        
                        if job_details:
                            enhanced_results.append(job_details)
                            seen_jobs.add(related_job['job_id'])
        
        return enhanced_results
