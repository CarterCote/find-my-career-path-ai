from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import BaseRetriever
from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text
from enum import Enum
from .job_graph import JobGraph
from ..models import JobPosting
import torch

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
        
        # Verify database has data
        count = self.db.execute(text("SELECT COUNT(*) FROM postings")).scalar()
        print(f"Database contains {count} job postings")  # Debug log
    
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
        try:
            print(f"Semantic search - Query: {query}, Filters: {filters}")  # Debug log
            
            # Clean the query string: remove newlines and extra spaces
            query = ' '.join(query.split())  # This converts all whitespace to single spaces
            print(f"Cleaned query: {query}")  # Debug log
            
            # Handle empty query
            if not query or query.isspace():
                query = "software engineer"  # Default fallback
            
            # Convert query to embedding - simplified encoding
            query_embedding = self.embed_model.encode(query) 
            print(f"Generated embedding shape: {query_embedding.shape}")  # Debug log
            
            sql = """
            WITH ranked_jobs AS (
                SELECT 
                    *,
                    1 - (description_embedding <=> :query_embedding::vector) AS semantic_score,
                    ts_rank(to_tsvector('english', description), plainto_tsquery('english', :query)) AS text_score
                FROM postings
                WHERE 1=1
            """
            
            params = {
                "query": query, 
                "query_embedding": query_embedding.tolist()  # Convert numpy array to list
            }
            
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
            
            results = self.db.execute(text(sql), params).fetchall()
            print(f"Found {len(results)} results in semantic search")  # Debug log
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            return []
    
    def sparse_search(self, query: str, filters: Dict = None):
        """Quick text-based search with hard constraints"""
        try:
            # Break down query into individual terms
            if filters and filters.get('required_skills'):
                required_terms = filters['required_skills']
                print(f"Searching for required terms: {required_terms}")
                
                # Construct WHERE clauses for each required term
                where_clauses = []
                params = {}
                
                for idx, term in enumerate(required_terms):
                    param_name = f"term_{idx}"
                    where_clauses.append(f"""
                        (
                            description ILIKE :%{param_name}% OR
                            title ILIKE :%{param_name}% OR
                            EXISTS (
                                SELECT 1 
                                FROM jsonb_array_elements_text(structured_description->'required_skills') skill
                                WHERE skill ILIKE :%{param_name}%
                            )
                        )
                    """)
                    params[param_name] = term
                
                sql = """
                SELECT 
                    *,
                    ts_rank(
                        to_tsvector('english', title || ' ' || description), 
                        to_tsquery('english', :query_terms)
                    ) AS relevance
                FROM postings
                WHERE 1=1
                AND {}
                """.format(' AND '.join(where_clauses))
                
                # Add other filters
                if filters.get('work_environment'):
                    sql += " AND structured_description->>'work_environment' ILIKE :work_env"
                    params['work_env'] = f"%{filters['work_environment']}%"
                
                sql += " ORDER BY relevance DESC LIMIT :limit"
                params.update({
                    'query_terms': ' | '.join(required_terms),
                    'limit': filters.get('limit', 100)
                })
                
                print(f"Executing sparse search with params: {params}")
                results = self.db.execute(text(sql), params).fetchall()
                print(f"Found {len(results)} results in sparse search")
                
                return results
                
        except Exception as e:
            print(f"Error in sparse search: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, filters: Dict = None):
        """Two-stage search: sparse then semantic"""
        try:
            # First get candidates that match hard constraints
            sparse_results = self.sparse_search(query, {"limit": 100, **filters})
            print(f"Sparse search found {len(sparse_results)} initial candidates")
            
            if not sparse_results:
                return []
            
            # Then refine with semantic search
            job_ids = [r.job_id for r in sparse_results]
            semantic_results = self.semantic_search(
                query, 
                {"job_ids": job_ids, "limit": 10, **filters}
            )
            
            return semantic_results
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            return []
    
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
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Required implementation of abstract method from BaseRetriever"""
        # Convert query to string if it's a QueryBundle
        query_str = query_bundle.query_str if isinstance(query_bundle, QueryBundle) else str(query_bundle)
        
        # Use our existing search method
        results = self.search_jobs(query_str)
        
        # Convert results to NodeWithScore format
        nodes_with_scores = []
        for result in results:
            # Create text content from job details
            text_content = f"""
            Title: {result.title}
            Company: {result.company_name}
            Description: {result.description}
            Location: {result.location}
            Salary Range: ${result.min_salary or 0:,.0f} - ${result.max_salary or 0:,.0f}
            """
            
            # Create node and add score
            node = TextNode(text=text_content)
            score = getattr(result, 'semantic_score', 0.0)
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        
        return nodes_with_scores
    
    def test_embedding(self, test_query: str = "software engineer"):
        """Test method to verify embedding functionality"""
        try:
            embedding = self.embed_model.encode([test_query])[0]
            print(f"Test embedding successful - Shape: {embedding.shape}")
            return True
        except Exception as e:
            print(f"Test embedding failed: {str(e)}")
            return False