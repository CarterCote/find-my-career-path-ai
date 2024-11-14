from typing import Dict, List, Set
import networkx as nx
from collections import defaultdict
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class JobGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.title_nodes = set()
        self.company_nodes = set()
        self.skill_nodes = set()
        # Initialize sentence transformer for semantic similarity
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def normalize_title(self, title: str) -> str:
        """Normalize job titles for better matching"""
        if not title:
            return "unknown"
        
        # Convert to lowercase and remove special characters
        normalized = title.lower().strip()
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        
        # Remove common words that don't add meaning
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of'}
        words = normalized.split()
        normalized = ' '.join(w for w in words if w not in stop_words)
        
        # Handle common title variations
        replacements = {
            'sr': 'senior',
            'jr': 'junior',
            'mgr': 'manager',
            'eng': 'engineer',
            'dev': 'developer',
            'admin': 'administrator',
            'assoc': 'associate'
        }
        
        words = normalized.split()
        normalized = ' '.join(replacements.get(w, w) for w in words)
        
        return normalized
    
    def build_graph(self, jobs_df):
        """Build enhanced graph from job postings data with structured descriptions"""
        print("Building job relationship graph...")
        
        # Verify required columns exist
        required_columns = ['job_id', 'company_name', 'title', 'med_salary', 'location', 'structured_description']
        missing_columns = [col for col in required_columns if col not in jobs_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # First pass: Create all nodes
        for _, job in jobs_df.iterrows():
            try:
                # Add job node with enriched attributes
                structured_desc = self._parse_structured_desc(job.structured_description)
                
                # Add job node
                self._add_job_node(job, structured_desc)
                
                # Add or get title node
                if pd.notna(job.title):
                    self._add_title_node(job)
                
                # Add or get company node
                if pd.notna(job.company_name):
                    self._add_company_node(job)
                
                # Add skill nodes from structured description
                if structured_desc:
                    self._add_skill_nodes(structured_desc)
                    
            except Exception as e:
                print(f"Error processing job {job.job_id}: {str(e)}")
                continue
            
        # Second pass: Create relationships
        self._create_relationships(jobs_df)
        
        print(f"Graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph
    
    def _parse_structured_desc(self, desc_str: str) -> Dict:
        """Safely parse structured description JSON"""
        if pd.isna(desc_str):
            return {}
        
        try:
            if isinstance(desc_str, str):
                # Remove any leading/trailing whitespace and quotes
                desc_str = desc_str.strip().strip('"\'')
                return json.loads(desc_str)
            elif isinstance(desc_str, dict):
                return desc_str
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
        except Exception as e:
            print(f"Unexpected error parsing description: {e}")
        
        return {}
    
    def _add_job_node(self, job, structured_desc: Dict):
        """Add job node with enriched attributes"""
        # Ensure structured_desc is a dictionary
        if not isinstance(structured_desc, dict):
            structured_desc = {}
        
        # Safely get values with defaults
        self.graph.add_node(
            f"job_{job.job_id}",
            type='job',
            title=str(job.title) if pd.notna(job.title) else "Unknown",
            company=str(job.company_name) if pd.notna(job.company_name) else "Unknown",
            salary=float(job.med_salary) if pd.notna(job.med_salary) else 0.0,
            location=str(job.location) if pd.notna(job.location) else "Unknown",
            required_skills=structured_desc.get('Required skills', []),  # Note the capital R
            experience_years=structured_desc.get('Years of experience', ''),  # Note the capital Y
            education=structured_desc.get('Education requirements', ''),  # Note the capital E
            work_environment=structured_desc.get('Work environment', '')  # Note the capital W
        )
    
    def _add_title_node(self, job):
        """Add title node with normalization"""
        normalized_title = self.normalize_title(job.title)
        if normalized_title not in self.title_nodes:
            self.graph.add_node(
                f"title_{normalized_title}",
                type='title',
                name=normalized_title
            )
            self.title_nodes.add(normalized_title)
        
        # Connect job to title
        self.graph.add_edge(
            f"job_{job.job_id}",
            f"title_{normalized_title}",
            relationship='has_title'
        )
    
    def _add_company_node(self, job):
        """Add company node"""
        if job.company_name not in self.company_nodes:
            self.graph.add_node(
                f"company_{job.company_name}",
                type='company',
                name=job.company_name
            )
            self.company_nodes.add(job.company_name)
        
        # Connect job to company
        self.graph.add_edge(
            f"job_{job.job_id}",
            f"company_{job.company_name}",
            relationship='posted_by'
        )
    
    def _add_skill_nodes(self, structured_desc: Dict):
        """Add skill nodes from structured description"""
        skills = structured_desc.get('required_skills', [])
        for skill in skills:
            skill = skill.lower().strip()
            if skill not in self.skill_nodes:
                self.graph.add_node(
                    f"skill_{skill}",
                    type='skill',
                    name=skill
                )
                self.skill_nodes.add(skill)
    
    def _create_relationships(self, jobs_df):
        """Create relationships between nodes"""
        # Connect related titles based on skill similarity
        title_nodes = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'title']
        
        for title1 in title_nodes:
            title1_jobs = self._get_jobs_for_title(title1)
            title1_skills = self._get_skills_for_jobs(title1_jobs)
            
            for title2 in title_nodes:
                if title1 != title2:
                    title2_jobs = self._get_jobs_for_title(title2)
                    title2_skills = self._get_skills_for_jobs(title2_jobs)
                    
                    # Calculate skill similarity
                    similarity = self._calculate_skill_similarity(title1_skills, title2_skills)
                    if similarity > 0.3:  # Threshold for relationship
                        self.graph.add_edge(
                            title1,
                            title2,
                            relationship='related_by_skills',
                            similarity=similarity
                        )
    
    def _get_jobs_for_title(self, title_node: str) -> List[str]:
        """Get all jobs with a given title"""
        return [n for n, _ in self.graph.edges(title_node) 
                if n.startswith('job_')]
    
    def _get_skills_for_jobs(self, job_nodes: List[str]) -> Set[str]:
        """Get all skills required by a list of jobs"""
        skills = set()
        for job in job_nodes:
            job_data = self.graph.nodes[job]
            skills.update(job_data.get('required_skills', []))
        return skills
    
    def _calculate_skill_similarity(self, skills1: Set[str], skills2: Set[str]) -> float:
        """Calculate similarity between two sets of skills using embeddings"""
        if not skills1 or not skills2:
            return 0.0
            
        # Get embeddings for both skill sets
        emb1 = self.encoder.encode(list(skills1))
        emb2 = self.encoder.encode(list(skills2))
        
        # Calculate average embeddings
        avg_emb1 = np.mean(emb1, axis=0)
        avg_emb2 = np.mean(emb2, axis=0)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            avg_emb1.reshape(1, -1),
            avg_emb2.reshape(1, -1)
        )[0][0]
        
        return float(similarity)