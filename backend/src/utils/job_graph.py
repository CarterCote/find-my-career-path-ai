from typing import Dict, List, Set
import networkx as nx
from collections import defaultdict

class JobGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.title_nodes = set()  # Track unique titles
        self.company_nodes = set()  # Track unique companies
        
    def build_graph(self, jobs_df):
        """Build graph from job postings data"""
        print("Building job relationship graph...")
        
        # First pass: Create nodes for titles and companies
        for _, job in jobs_df.iterrows():
            # Add job node
            self.graph.add_node(
                f"job_{job.job_id}",
                type='job',
                title=job.title,
                company=job.company_name,
                salary=job.med_salary,
                location=job.location
            )
            
            # Add or get title node
            normalized_title = self.normalize_title(job.title)
            if normalized_title not in self.title_nodes:
                self.graph.add_node(
                    f"title_{normalized_title}",
                    type='title',
                    name=normalized_title
                )
                self.title_nodes.add(normalized_title)
            
            # Add or get company node
            if job.company_name not in self.company_nodes:
                self.graph.add_node(
                    f"company_{job.company_name}",
                    type='company',
                    name=job.company_name
                )
                self.company_nodes.add(job.company_name)
            
            # Connect job to its title and company
            self.graph.add_edge(
                f"job_{job.job_id}",
                f"title_{normalized_title}",
                relationship='has_title'
            )
            self.graph.add_edge(
                f"job_{job.job_id}",
                f"company_{job.company_name}",
                relationship='posted_by'
            )
        
        # Second pass: Connect related titles
        self._connect_related_titles()
        
        print(f"Graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph
    
    def normalize_title(self, title: str) -> str:
        """Normalize job titles (e.g., 'Senior SWE' -> 'Software Engineer')"""
        title = title.lower()
        
        # Add your title normalization logic here
        # Example:
        replacements = {
            'swe': 'software engineer',
            'sr.': 'senior',
            'jr.': 'junior'
        }
        
        for old, new in replacements.items():
            title = title.replace(old, new)
        
        return title.strip()
    
    def _connect_related_titles(self):
        """Connect related job titles based on similarity"""
        title_nodes = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'title']
        
        for title1 in title_nodes:
            for title2 in title_nodes:
                if title1 != title2:
                    # Add similarity logic here
                    # Example: Levenshtein distance or embedding similarity
                    if self._are_titles_related(title1, title2):
                        self.graph.add_edge(
                            title1,
                            title2,
                            relationship='related_to'
                        )
    
    def _are_titles_related(self, title1: str, title2: str) -> bool:
        """Determine if two titles are related"""
        # Add your similarity logic here
        # Example: "Senior Software Engineer" is related to "Software Engineer"
        return False  # Placeholder
    
    def get_related_jobs(self, job_id: str, max_distance: int = 2) -> List[Dict]:
        """Get related jobs within n-degrees of separation"""
        if f"job_{job_id}" not in self.graph:
            return []
            
        related_jobs = []
        for node in nx.single_source_shortest_path_length(
            self.graph, f"job_{job_id}", cutoff=max_distance
        ):
            if node.startswith('job_') and node != f"job_{job_id}":
                job_data = self.graph.nodes[node]
                related_jobs.append({
                    'job_id': node.replace('job_', ''),
                    'title': job_data['title'],
                    'company': job_data['company'],
                    'relationship_distance': nx.shortest_path_length(
                        self.graph, f"job_{job_id}", node
                    )
                })
        
        return sorted(related_jobs, key=lambda x: x['relationship_distance'])