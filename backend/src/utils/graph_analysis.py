import networkx as nx
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import json
import pandas as pd
import csv
import sys
import os

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from backend.src.utils.job_graph import JobGraph

def generate_graph_from_csv(csv_path: str, output_pkl_path: str = None):
    """Generate a graph from a processed CSV file with structured descriptions"""
    print(f"Loading data from {csv_path}...")
    
    # Load the processed CSV
    df = pd.read_csv(
        csv_path,
        low_memory=False,
        dtype={
            'job_id': str,
            'company_name': str,
            'title': str,
            'location': str,
            'company_id': str,
            'structured_description': str
        }
    )
    
    print(f"Loaded {len(df)} records")
    
    # Debug: Check the structured_description column
    print("\nChecking structured_description column:")
    print(f"Data type: {df['structured_description'].dtype}")
    print("\nSample structured description:")
    sample_desc = df.loc[df['structured_description'].notna()].iloc[0]['structured_description']
    print(sample_desc)
    
    # Initialize and build graph
    job_graph = JobGraph()
    graph = job_graph.build_graph(df)
    
    # Save the graph
    if output_pkl_path is None:
        output_pkl_path = 'backend/data/job_graph.pkl'
        
    print(f"\nSaving graph to {output_pkl_path}")
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(graph, f)
    
    return graph

def analyze_graph(pkl_path: str = 'backend/data/job_graph.pkl'):
    # Load the saved graph
    with open(pkl_path, 'rb') as f:
        G = pickle.load(f)
    
    # Print basic statistics
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Count node types
    node_types = {}
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\nNode types distribution:")
    for node_type, count in node_types.items():
        print(f"{node_type}: {count}")

if __name__ == "__main__":
    # Example usage:
    checkpoint_path = 'backend/data/checkpoints/enriched_data_checkpoint_2900.csv'
    
    # Generate graph from checkpoint
    G = generate_graph_from_csv(checkpoint_path)
    
    # Analyze the generated graph
    analyze_graph() 