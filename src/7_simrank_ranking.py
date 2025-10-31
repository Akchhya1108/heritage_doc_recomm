import json
import os
import numpy as np
import networkx as nx
import pickle
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

# Directories
KG_DIR = "knowledge_graph"
RANKING_DIR = "data/ranking"

# Files
INTEGRATED_KG_FILE = os.path.join(KG_DIR, "heritage_kg_integrated.gpickle")
SIMRANK_FILE = os.path.join(RANKING_DIR, "simrank_matrix.npy")
SIMRANK_MAPPING_FILE = os.path.join(RANKING_DIR, "simrank_mapping.json")
RANKING_STATS_FILE = os.path.join(RANKING_DIR, "ranking_statistics.json")

# SimRank parameters
SIMRANK_C = 0.8  # Decay factor (how much to trust indirect connections)
SIMRANK_MAX_ITER = 10  # Number of iterations
SIMRANK_EPSILON = 1e-4  # Convergence threshold

# ========== SIMRANK ALGORITHM ==========

def compute_simrank(G, doc_nodes, C=0.8, max_iter=10, epsilon=1e-4):
    """
    Compute SimRank similarity between document nodes
    
    SimRank: Two objects are similar if they are related to similar objects
    S(a,b) = C / (|I(a)| * |I(b)|) * Σ S(I_i(a), I_j(b))
    where I(x) = in-neighbors of x
    """
    print("\n[Phase 1] Computing SimRank similarity...")
    print(f"  Configuration: C={C}, max_iter={max_iter}, epsilon={epsilon}")
    
    n = len(doc_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(doc_nodes)}
    
    # Initialize similarity matrix
    # S[i,i] = 1 (a node is completely similar to itself)
    # S[i,j] = 0 for i != j (initially no similarity)
    S = np.zeros((n, n))
    np.fill_diagonal(S, 1.0)
    
    print(f"  Matrix size: {n}x{n} documents")
    
    # Precompute in-neighbors for efficiency
    in_neighbors = {}
    for node in doc_nodes:
        in_neighbors[node] = list(G.predecessors(node)) if G.is_directed() else list(G.neighbors(node))
    
    # Iterative computation
    for iteration in range(max_iter):
        S_prev = S.copy()
        max_change = 0.0
        
        # Update similarity for each pair
        for i, node_a in enumerate(doc_nodes):
            for j in range(i+1, n):  # Only upper triangle (symmetric)
                node_b = doc_nodes[j]
                
                # Get in-neighbors
                I_a = in_neighbors[node_a]
                I_b = in_neighbors[node_b]
                
                if not I_a or not I_b:
                    # If either has no in-neighbors, similarity is 0
                    S[i, j] = 0.0
                else:
                    # SimRank formula
                    sum_sim = 0.0
                    for neighbor_a in I_a:
                        if neighbor_a not in node_to_idx:
                            continue
                        idx_a = node_to_idx[neighbor_a]
                        
                        for neighbor_b in I_b:
                            if neighbor_b not in node_to_idx:
                                continue
                            idx_b = node_to_idx[neighbor_b]
                            
                            sum_sim += S_prev[idx_a, idx_b]
                    
                    S[i, j] = (C / (len(I_a) * len(I_b))) * sum_sim
                
                # Symmetric
                S[j, i] = S[i, j]
                
                # Track convergence
                change = abs(S[i, j] - S_prev[i, j])
                max_change = max(max_change, change)
        
        print(f"  Iteration {iteration+1}/{max_iter}: max_change = {max_change:.6f}")
        
        # Check convergence
        if max_change < epsilon:
            print(f"  ✓ Converged after {iteration+1} iterations")
            break
    
    print(f"  ✓ SimRank computation complete")
    print(f"  Similarity range: [{S[S != 1.0].min():.6f}, {S[S != 1.0].max():.6f}]")
    print(f"  Mean similarity: {S[S != 1.0].mean():.6f}")
    
    return S, node_to_idx

# ========== HORN'S INDEX ==========

def compute_horns_index(G, doc_nodes):
    """
    Compute Horn's Index for document importance
    
    Horn's Index combines:
    - Degree centrality (how connected)
    - PageRank (importance in network)
    - Clustering coefficient (local cohesion)
    - Betweenness centrality (bridging role)
    """
    print("\n[Phase 2] Computing Horn's Index...")
    
    horns_index = {}
    
    # Get precomputed metrics
    pagerank = {node: G.nodes[node].get('pagerank', 0.0) for node in doc_nodes}
    degree_cent = {node: G.nodes[node].get('degree_centrality', 0.0) for node in doc_nodes}
    
    # Compute clustering coefficient
    print("  Computing clustering coefficients...")
    clustering = nx.clustering(G)
    
    # Compute betweenness for document subgraph (expensive, so use sample)
    print("  Computing betweenness centrality...")
    doc_subgraph = G.subgraph(doc_nodes)
    
    if len(doc_nodes) <= 200:
        betweenness = nx.betweenness_centrality(doc_subgraph)
    else:
        # Use approximate betweenness for large graphs
        betweenness = nx.betweenness_centrality(doc_subgraph, k=200)
    
    # Combine metrics into Horn's Index
    # Horn's Index = α*PageRank + β*Degree + γ*Clustering + δ*Betweenness
    # Weights tuned for heritage documents
    alpha, beta, gamma, delta = 0.4, 0.3, 0.15, 0.15
    
    for node in doc_nodes:
        pr = pagerank.get(node, 0.0)
        dc = degree_cent.get(node, 0.0)
        cc = clustering.get(node, 0.0)
        bc = betweenness.get(node, 0.0)
        
        # Normalize and combine
        horns_index[node] = alpha * pr + beta * dc + gamma * cc + delta * bc
    
    # Normalize to [0, 1]
    max_horn = max(horns_index.values()) if horns_index else 1.0
    if max_horn > 0:
        horns_index = {k: v/max_horn for k, v in horns_index.items()}
    
    print(f"  ✓ Horn's Index computed for {len(horns_index)} documents")
    print(f"  Range: [{min(horns_index.values()):.6f}, {max(horns_index.values()):.6f}]")
    
    # Top documents by Horn's Index
    top_horns = sorted(horns_index.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  📊 Top 10 documents by Horn's Index:")
    for node, score in top_horns:
        title = G.nodes[node].get('title', node)
        print(f"     {title[:50]}: {score:.6f}")
    
    return horns_index

# ========== COMBINED RANKING ==========

def compute_combined_ranking(simrank_matrix, horns_index, doc_nodes, node_to_idx, 
                            query_doc_idx, top_k=20, lambda_param=0.6):
    """
    Combine SimRank and Horn's Index for final ranking
    
    Final_Score(d) = λ * SimRank(query, d) + (1-λ) * Horn's_Index(d)
    
    Args:
        lambda_param: Weight for SimRank (0.6 means 60% SimRank, 40% Horn's)
    """
    scores = []
    
    for idx, node in enumerate(doc_nodes):
        if idx == query_doc_idx:
            continue  # Don't recommend the query document itself
        
        # SimRank similarity to query
        simrank_score = simrank_matrix[query_doc_idx, idx]
        
        # Horn's Index importance
        horn_score = horns_index.get(node, 0.0)
        
        # Combined score
        final_score = lambda_param * simrank_score + (1 - lambda_param) * horn_score
        
        scores.append((node, simrank_score, horn_score, final_score))
    
    # Sort by final score
    scores.sort(key=lambda x: x[3], reverse=True)
    
    return scores[:top_k]

# ========== EVALUATION METRICS ==========

def evaluate_ranking_quality(G, simrank_matrix, horns_index, doc_nodes, node_to_idx):
    """Evaluate ranking quality with metrics"""
    print("\n[Phase 3] Evaluating ranking quality...")
    
    # Sample 50 documents as queries
    import random
    sample_size = min(50, len(doc_nodes))
    query_docs = random.sample(range(len(doc_nodes)), sample_size)
    
    # Metrics
    diversity_scores = []
    cluster_precision = []
    domain_coverage = []
    
    for query_idx in query_docs:
        query_node = doc_nodes[query_idx]
        query_cluster = G.nodes[query_node].get('cluster_id', -1)
        query_domains = set(G.nodes[query_node].get('domains', []))
        
        # Get top-10 recommendations
        recommendations = compute_combined_ranking(
            simrank_matrix, horns_index, doc_nodes, node_to_idx, query_idx, top_k=10
        )
        
        # 1. Cluster Precision: How many recommendations are from same cluster?
        same_cluster = sum(1 for node, _, _, _ in recommendations 
                          if G.nodes[node].get('cluster_id') == query_cluster)
        cluster_precision.append(same_cluster / 10.0)
        
        # 2. Domain Coverage: How many different domains are covered?
        rec_domains = set()
        for node, _, _, _ in recommendations:
            rec_domains.update(G.nodes[node].get('domains', []))
        domain_coverage.append(len(rec_domains))
        
        # 3. Diversity: Average pairwise dissimilarity in recommendations
        if len(recommendations) > 1:
            rec_indices = [node_to_idx[node] for node, _, _, _ in recommendations if node in node_to_idx]
            pairwise_sim = []
            for i in range(len(rec_indices)):
                for j in range(i+1, len(rec_indices)):
                    pairwise_sim.append(simrank_matrix[rec_indices[i], rec_indices[j]])
            
            diversity = 1.0 - np.mean(pairwise_sim) if pairwise_sim else 0.0
            diversity_scores.append(diversity)
    
    metrics = {
        'cluster_precision': {
            'mean': float(np.mean(cluster_precision)),
            'std': float(np.std(cluster_precision))
        },
        'domain_coverage': {
            'mean': float(np.mean(domain_coverage)),
            'std': float(np.std(domain_coverage))
        },
        'diversity': {
            'mean': float(np.mean(diversity_scores)),
            'std': float(np.std(diversity_scores))
        }
    }
    
    print(f"  📊 Ranking Quality Metrics:")
    print(f"     Cluster Precision: {metrics['cluster_precision']['mean']:.3f} ± {metrics['cluster_precision']['std']:.3f}")
    print(f"     Domain Coverage: {metrics['domain_coverage']['mean']:.2f} ± {metrics['domain_coverage']['std']:.2f}")
    print(f"     Diversity Score: {metrics['diversity']['mean']:.3f} ± {metrics['diversity']['std']:.3f}")
    
    return metrics

# ========== VISUALIZATION ==========

def visualize_ranking_example(G, simrank_matrix, horns_index, doc_nodes, node_to_idx):
    """Create visualization showing a ranking example"""
    print("\n[Phase 4] Creating ranking visualization...")
    
    # Pick a random query document
    query_idx = len(doc_nodes) // 2  # Middle document
    query_node = doc_nodes[query_idx]
    query_title = G.nodes[query_node].get('title', 'Unknown')
    
    # Get top-10 recommendations
    recommendations = compute_combined_ranking(
        simrank_matrix, horns_index, doc_nodes, node_to_idx, query_idx, top_k=10
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Score breakdown
    ax1 = axes[0]
    rec_titles = [G.nodes[node].get('title', '')[:30] + '...' for node, _, _, _ in recommendations]
    simrank_scores = [sr for _, sr, _, _ in recommendations]
    horn_scores = [hr for _, _, hr, _ in recommendations]
    final_scores = [fs for _, _, _, fs in recommendations]
    
    x = np.arange(len(rec_titles))
    width = 0.25
    
    ax1.barh(x - width, simrank_scores, width, label='SimRank', alpha=0.8, color='#3498db')
    ax1.barh(x, horn_scores, width, label="Horn's Index", alpha=0.8, color='#e74c3c')
    ax1.barh(x + width, final_scores, width, label='Combined Score', alpha=0.8, color='#2ecc71')
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(rec_titles, fontsize=8)
    ax1.set_xlabel('Score', fontsize=10)
    ax1.set_title(f'Top-10 Recommendations for:\n"{query_title[:50]}..."', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='x', alpha=0.3)
    
    # Right plot: Similarity distribution
    ax2 = axes[1]
    all_similarities = simrank_matrix[query_idx, :]
    all_similarities = all_similarities[all_similarities != 1.0]  # Remove self-similarity
    
    ax2.hist(all_similarities, bins=50, alpha=0.7, color='#9b59b6', edgecolor='black')
    ax2.axvline(np.mean(simrank_scores), color='#e74c3c', linestyle='--', linewidth=2, 
                label=f'Mean of top-10: {np.mean(simrank_scores):.3f}')
    ax2.set_xlabel('SimRank Similarity', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('SimRank Similarity Distribution', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    viz_file = os.path.join(RANKING_DIR, 'ranking_example.png')
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Visualization saved to: {viz_file}")
    plt.close()

# ========== MAIN EXECUTION ==========

def load_knowledge_graph():
    """Load integrated knowledge graph"""
    print("\n[Loading] Integrated knowledge graph...")
    
    with open(INTEGRATED_KG_FILE, 'rb') as f:
        G = pickle.load(f)
    
    print(f"✓ Loaded KG with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G

def save_ranking_data(simrank_matrix, node_to_idx, horns_index, doc_nodes, G, metrics):
    """Save ranking matrices and mappings"""
    print("\n[Phase 5] Saving ranking data...")
    
    os.makedirs(RANKING_DIR, exist_ok=True)
    
    # Save SimRank matrix
    np.save(SIMRANK_FILE, simrank_matrix)
    print(f"  ✓ SimRank matrix saved: {SIMRANK_FILE}")
    
    # Save mapping
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    mapping = {
        'doc_nodes': doc_nodes,
        'node_to_idx': node_to_idx,
        'idx_to_node': idx_to_node,
        'document_info': []
    }
    
    for idx, node in enumerate(doc_nodes):
        mapping['document_info'].append({
            'index': idx,
            'node_id': node,
            'title': G.nodes[node].get('title', ''),
            'cluster_id': G.nodes[node].get('cluster_id', -1),
            'cluster_label': G.nodes[node].get('cluster_label', ''),
            'domains': G.nodes[node].get('domains', []),
            'heritage_types': G.nodes[node].get('heritage_types', []),
            'horns_index': horns_index.get(node, 0.0)
        })
    
    with open(SIMRANK_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Mapping saved: {SIMRANK_MAPPING_FILE}")
    
    # Save statistics
    stats = {
        'num_documents': len(doc_nodes),
        'simrank_config': {
            'C': SIMRANK_C,
            'max_iterations': SIMRANK_MAX_ITER,
            'epsilon': SIMRANK_EPSILON
        },
        'simrank_statistics': {
            'mean': float(simrank_matrix[simrank_matrix != 1.0].mean()),
            'std': float(simrank_matrix[simrank_matrix != 1.0].std()),
            'min': float(simrank_matrix[simrank_matrix != 1.0].min()),
            'max': float(simrank_matrix[simrank_matrix != 1.0].max())
        },
        'horns_index_statistics': {
            'mean': float(np.mean(list(horns_index.values()))),
            'std': float(np.std(list(horns_index.values()))),
            'min': float(min(horns_index.values())),
            'max': float(max(horns_index.values()))
        },
        'ranking_quality_metrics': metrics,
        'computation_date': datetime.now().isoformat()
    }
    
    with open(RANKING_STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Statistics saved: {RANKING_STATS_FILE}")

def main():
    print("="*70)
    print("SIMRANK + HORN'S INDEX RANKING SYSTEM")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load KG
    G = load_knowledge_graph()
    
    # Get all document nodes
    doc_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'document']
    print(f"\n📄 Processing {len(doc_nodes)} documents")
    
    # Compute SimRank
    simrank_matrix, node_to_idx = compute_simrank(G, doc_nodes, 
                                                   C=SIMRANK_C, 
                                                   max_iter=SIMRANK_MAX_ITER, 
                                                   epsilon=SIMRANK_EPSILON)
    
    # Compute Horn's Index
    horns_index = compute_horns_index(G, doc_nodes)
    
    # Evaluate ranking quality
    metrics = evaluate_ranking_quality(G, simrank_matrix, horns_index, doc_nodes, node_to_idx)
    
    # Visualize example
    visualize_ranking_example(G, simrank_matrix, horns_index, doc_nodes, node_to_idx)
    
    # Save everything
    save_ranking_data(simrank_matrix, node_to_idx, horns_index, doc_nodes, G, metrics)
    
    # Summary
    print("\n" + "="*70)
    print("RANKING SYSTEM COMPLETE")
    print("="*70)
    print(f"✅ Computed SimRank for {len(doc_nodes)} documents")
    print(f"✅ Computed Horn's Index for all documents")
    print(f"✅ Matrix size: {simrank_matrix.shape}")
    print(f"\n📊 Ranking Quality:")
    print(f"   Cluster Precision: {metrics['cluster_precision']['mean']:.3f}")
    print(f"   Domain Coverage: {metrics['domain_coverage']['mean']:.2f} domains")
    print(f"   Diversity: {metrics['diversity']['mean']:.3f}")
    print(f"\n💾 Files created:")
    print(f"   - {SIMRANK_FILE}")
    print(f"   - {SIMRANK_MAPPING_FILE}")
    print(f"   - {RANKING_STATS_FILE}")
    print(f"\n⏱ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()