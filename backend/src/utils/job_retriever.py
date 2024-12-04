from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import BaseRetriever
from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text, bindparam
from enum import Enum
from .job_graph import JobGraph
from ..models import JobPosting
import torch
from sqlalchemy.types import ARRAY, Float, String, Integer

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
        """Two-stage retrieval: sparse then dense"""
        try:
            print("\n========== TWO-STAGE RETRIEVAL DEBUG ==========")
            filters = filters or {}
            
            # Stage 1: Sparse Retrieval (Quick Filtering)
            print("1. Stage 1 - Sparse Retrieval (Hard Constraints)")
            sparse_results = self.sparse_search(
                query,
                filters={
                    'skills': filters.get('skills', []),
                    'work_culture': filters.get('work_culture', []),
                    'core_values': filters.get('core_values', []),
                    'limit': 1000  # Larger pool for semantic search
                }
            )
            print(f"Found {len(sparse_results)} candidates after sparse filtering")
            
            if not sparse_results:
                return []
                
            # Stage 2: Dense Retrieval using semantic_search
            print("\n2. Stage 2 - Dense Retrieval (Semantic Search)")
            job_ids = [job['job_id'] for job in sparse_results]
            
            semantic_results = self.semantic_search(
                query,
                filters={
                    'job_ids': job_ids,  # Only search within sparse results
                    'limit': 100  # Return 100 for QA evaluation
                }
            )
            
            print(f"\nReturning top {len(semantic_results)} semantically matched results for QA evaluation")
            return semantic_results
            
        except Exception as e:
            print(f"Error in search_jobs: {str(e)}")
            return []
    
    def semantic_search(self, query: str, filters: Dict = None):
        try:
            filters = filters or {}
            
            # Clean the query string
            query = ' '.join(query.split())
            query = query.replace("'", "''")  # Escape single quotes
            
            # Handle empty query
            if not query or query.isspace():
                query = "software engineer"
            
            # Convert query to embedding
            query_embedding = self.embed_model.encode(query)
            
            # Build the base SQL using bindparams for vector comparison
            sql = """
            WITH ranked_jobs AS (
                SELECT 
                    job_id,
                    title,
                    company_name,
                    description,
                    location,
                    min_salary,
                    max_salary,
                    1 - (description_embedding <=> array_to_vector(:query_embedding)) AS semantic_score,
                    ts_rank(to_tsvector('english', description), plainto_tsquery('english', :query)) AS text_score
                FROM postings
                WHERE 1=1
            """
            
            params = {
                'query_embedding': query_embedding.tolist(),  # Convert numpy array to list
                'query': query,
                'limit': filters.get('limit', 100)
            }
            
            # Add job_ids filter if present
            if 'job_ids' in filters and filters['job_ids']:
                sql += " AND job_id = ANY(:job_ids)"
                params['job_ids'] = filters['job_ids']
            
            # Complete the query
            sql += """
                )
                SELECT *
                FROM ranked_jobs
                ORDER BY (semantic_score + text_score) DESC
                LIMIT :limit
            """
            
            # Create SQLAlchemy text object with parameters
            stmt = text(sql)
            for key, value in params.items():
                stmt = stmt.bindparams(bindparam(key, value))
            
            results = self.db.execute(stmt).fetchall()
            formatted_results = self._format_results(results)
            
            # Add title relevance scores
            for job in formatted_results:
                title_score = self._check_title_relevance(job['title'], query)
                job['title_match_score'] = title_score
                # Combine semantic, text, and title scores
                job['relevance_score'] = (
                    job.get('semantic_score', 0.0) + 
                    job.get('text_score', 0.0) + 
                    title_score
                ) / 3
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return []
    
    def _format_results(self, results):
        """Format database results into a consistent structure"""
        formatted = []
        for row in results:
            try:
                # Convert row to dict safely
                if hasattr(row, '_mapping'):
                    # SQLAlchemy 2.0+ Row object
                    job_dict = dict(row._mapping)
                elif hasattr(row, 'keys'):
                    # SQLAlchemy Result object
                    job_dict = dict(zip(row.keys(), row))
                else:
                    # Regular tuple
                    job_dict = dict(row)
                
                # Remove embedding from response (it's large and not needed)
                job_dict.pop('description_embedding', None)
                
                # Convert scores to float with safe default values
                semantic_score = job_dict.get('semantic_score')
                text_score = job_dict.get('text_score')
                
                job_dict['relevance_score'] = float(semantic_score) if semantic_score is not None else 0.0
                job_dict['text_match_score'] = float(text_score) if text_score is not None else 0.0
                
                # Clean up the original score fields
                job_dict.pop('semantic_score', None)
                job_dict.pop('text_score', None)
                
                formatted.append(job_dict)
                
            except Exception as e:
                print(f"Error formatting result row: {str(e)}")
                print(f"Row type: {type(row)}")
                print(f"Row content: {row}")
                continue
        
        return formatted
    
    def sparse_search(self, query: str, filters: Dict = None):
        """Perform sparse (keyword-based) search"""
        try:
            print("\n========== SPARSE SEARCH DEBUG ==========")
            filters = filters or {}
            
            # Helper function to format terms for tsquery
            def format_for_tsquery(terms):
                formatted_terms = []
                for term in terms:
                    # Replace spaces and special chars with underscores
                    clean_term = term.replace('/', ' ').replace('-', ' ')
                    # Split into words and join with |
                    words = [word.strip() for word in clean_term.split() if word.strip()]
                    formatted_terms.extend(words)
                # Join all terms with | (OR)
                return ' | '.join(formatted_terms)
            
            # Format terms for each category
            skills_terms = format_for_tsquery(filters.get('skills', []))
            culture_terms = format_for_tsquery(filters.get('work_culture', []))
            values_terms = format_for_tsquery(filters.get('core_values', []))
            
            print(f"1. Filter terms:")
            print(f"Skills: {skills_terms}")
            print(f"Culture: {culture_terms}")
            print(f"Values: {values_terms}")
            
            # Build SQL query with more flexible matching
            sql = """
            WITH job_matches AS (
                SELECT 
                    p.job_id,
                    p.title,
                    p.company_name,
                    p.description,
                    p.location,
                    p.min_salary,
                    p.max_salary,
                    ts_rank(to_tsvector('english', p.description || ' ' || p.title), 
                           to_tsquery('english', :skills_terms)) as skills_score,
                    ts_rank(to_tsvector('english', p.description || ' ' || p.title), 
                           to_tsquery('english', :culture_terms)) as culture_score,
                    ts_rank(to_tsvector('english', p.description || ' ' || p.title), 
                           to_tsquery('english', :values_terms)) as values_score
                FROM postings p
                WHERE 
                    -- Require some minimum matching in each category
                    ts_rank(to_tsvector('english', p.description || ' ' || p.title), 
                           to_tsquery('english', :skills_terms)) > 0.01
                    AND ts_rank(to_tsvector('english', p.description || ' ' || p.title), 
                               to_tsquery('english', :culture_terms)) > 0.01
                    AND ts_rank(to_tsvector('english', p.description || ' ' || p.title), 
                               to_tsquery('english', :values_terms)) > 0.01
            )
            SELECT 
                job_id,
                title,
                company_name,
                description,
                location,
                min_salary,
                max_salary,
                (skills_score + culture_score + values_score) as text_score,
                skills_score,
                culture_score,
                values_score
            FROM job_matches
            ORDER BY text_score DESC
            LIMIT :limit
            """
            
            params = {
                'skills_terms': skills_terms,
                'culture_terms': culture_terms,
                'values_terms': values_terms,
                'limit': filters.get('limit', 100)
            }
            
            print(f"2. Executing sparse search with category-specific filters...")
            results = self.db.execute(text(sql), params).fetchall()
            print(f"3. Found {len(results)} results matching ALL filter criteria")
            
            # Convert results to list of dicts
            formatted_results = []
            for row in results:
                job_dict = {
                    'job_id': row.job_id,
                    'title': row.title,
                    'company_name': row.company_name,
                    'description': row.description,
                    'location': row.location,
                    'min_salary': row.min_salary,
                    'max_salary': row.max_salary,
                    'relevance_score': float(row.text_score),
                    'scores': {
                        'skills': float(row.skills_score),
                        'culture': float(row.culture_score),
                        'values': float(row.values_score)
                    }
                }
                formatted_results.append(job_dict)
                
                print(f"\nJob: {job_dict['title']}")
                print(f"Skills Score: {job_dict['scores']['skills']:.3f}")
                print(f"Culture Score: {job_dict['scores']['culture']:.3f}")
                print(f"Values Score: {job_dict['scores']['values']:.3f}")
                print(f"Total Score: {job_dict['relevance_score']:.3f}")
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in sparse search: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
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
        try:
            # First get base results from semantic/sparse search
            base_results = self.semantic_search(query, filters)
            
            # Get related jobs through graph traversal
            enhanced_results = []
            seen_jobs = set()
            
            for job in base_results:
                job_id = job.get('job_id')  # Use dict access instead of attribute
                if not job_id or job_id in seen_jobs:
                    continue
                    
                # Add the base job
                enhanced_results.append(job)
                seen_jobs.add(job_id)
                
                # Find related jobs through title and company connections
                if hasattr(self, 'graph'):
                    related_jobs = self.job_graph.get_related_jobs(job_id, max_distance=2)
                    for related_job in related_jobs:
                        related_id = related_job.get('job_id')
                        if related_id and related_id not in seen_jobs:
                            # Get full job details from database
                            job_details = self.db.query(JobPosting).filter(
                                JobPosting.job_id == related_id
                            ).first()
                            
                            if job_details:
                                enhanced_results.append(job_details)
                                seen_jobs.add(related_id)
            
            return enhanced_results
                
        except Exception as e:
            print(f"Error in graph search: {str(e)}")
            return base_results  # Fall back to base results on error
    
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
    
    def _check_title_relevance(self, title: str, query: str) -> float:
        """Check how relevant a job title is to the search query"""
        try:
            # Clean and normalize strings
            title = title.lower()
            query = query.lower()
            
            # Split into words
            title_words = set(title.split())
            query_words = set(query.split())
            
            # Calculate overlap
            common_words = title_words.intersection(query_words)
            if not query_words:
                return 0.0
            
            # Return percentage of query words found in title
            return len(common_words) / len(query_words)
            
        except Exception as e:
            print(f"Error checking title relevance: {str(e)}")
            return 0.0