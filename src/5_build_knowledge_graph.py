import json
import os
import networkx as nx
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Download WordNet data if needed
try:
    wn.synsets('test')
except:
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Directories
CLASSIFIED_DIR = "data/classified"
KG_DIR = "knowledge_graph"
EMBEDDINGS_DIR = "data/embeddings"

# Files
CLASSIFIED_FILE = os.path.join(CLASSIFIED_DIR, "classified_documents.json")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "document_embeddings.npy")
KG_FILE = os.path.join(KG_DIR, "heritage_kg.gpickle")
KG_STATS_FILE = os.path.join(KG_DIR, "kg_statistics.json")
KG_VIZ_FILE = os.path.join(KG_DIR, "kg_visualization.png")

# ========== CONFIGURATION ==========
CONFIG = {
    'similarity_threshold': 0.6,  # Cosine similarity threshold for doc-doc edges
    'concept_similarity_threshold': 0.3,  # LOWERED from 0.5 for Lesk
    'top_k_cluster_connections': 10,  # Only connect to top-K similar docs in cluster
    'max_entities_per_doc': 5,  # Max entities to extract per document
}

# ========== LESK SIMILARITY (Improved) ==========

def lesk_similarity(word1, word2):
    """
    Compute semantic similarity using Lesk algorithm
    """
    try:
        # Clean inputs
        word1_clean = word1.lower().replace(' ', '_').replace('-', '_')
        word2_clean = word2.lower().replace(' ', '_').replace('-', '_')
        
        # Get synsets for both words
        synsets1 = wn.synsets(word1_clean)
        synsets2 = wn.synsets(word2_clean)
        
        if not synsets1 or not synsets2:
            return 0.0
        
        # Find maximum similarity between any pair of synsets
        max_sim = 0.0
        for s1 in synsets1[:5]:  # Increased from 3 to 5 senses
            for s2 in synsets2[:5]:
                try:
                    # Use path similarity (0 to 1)
                    sim = s1.path_similarity(s2)
                    if sim and sim > max_sim:
                        max_sim = sim
                except:
                    continue
        
        return max_sim
    
    except Exception as e:
        return 0.0

def compute_concept_similarity(concept1, concept2):
    """Compute similarity between two concepts with domain knowledge"""
    
    # Direct match
    if concept1.lower() == concept2.lower():
        return 1.0
    
    # Predefined related concepts (domain knowledge)
    related_groups = [
        # Religious structures
        {'temple', 'mosque', 'church', 'monastery', 'shrine', 'cathedral', 'stupa', 'pagoda'},
        # Military structures
        {'fort', 'fortress', 'citadel', 'castle', 'stronghold', 'garrison'},
        # Royal/residential
        {'palace', 'mansion', 'haveli', 'royal_residence'},
        # Time periods
        {'ancient', 'medieval', 'modern', 'prehistoric', 'contemporary'},
        # Commemorative
        {'monument', 'memorial', 'statue', 'cenotaph', 'tomb'},
        # Cultural/artistic
        {'art', 'craft', 'painting', 'sculpture', 'architecture'},
        # Performance
        {'dance', 'music', 'theater', 'performance', 'ritual'},
        # Literature
        {'literature', 'poetry', 'manuscript', 'text', 'epic'},
        # Regions
        {'north', 'south', 'east', 'west', 'central'},
    ]
    
    # Check if both concepts are in the same semantic group
    for group in related_groups:
        c1_lower = concept1.lower().replace(' ', '_')
        c2_lower = concept2.lower().replace(' ', '_')
        
        if c1_lower in group and c2_lower in group:
            return 0.8  # High similarity for same group
    
    # Use Lesk similarity as fallback
    lesk_sim = lesk_similarity(concept1, concept2)
    
    return lesk_sim

# ========== GRAPH CONSTRUCTION ==========

def load_data():
    """Load classified documents and embeddings"""
    print("\n[Phase 1] Loading data...")
    
    with open(CLASSIFIED_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"✓ Loaded {len(documents)} classified documents")
    
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"✓ Loaded embeddings: {embeddings.shape}")
    
    return documents, embeddings

def create_base_graph(documents):
    """Create base knowledge graph with document nodes"""
    print("\n[Phase 2] Creating base graph structure...")
    
    G = nx.Graph()
    
    # Add document nodes
    for idx, doc in enumerate(documents):
        G.add_node(
            f"doc_{idx}",
            node_type='document',
            title=doc['title'],
            cluster_id=doc['cluster_id'],
            cluster_label=doc['cluster_label'],
            heritage_types=doc['classifications']['heritage_types'],
            domains=doc['classifications']['domains'],
            time_period=doc['classifications']['time_period'],
            region=doc['classifications']['region'],
            source=doc['source']
        )
    
    print(f"  ✓ Added {len(documents)} document nodes")
    
    return G

def add_entity_nodes(G, documents):
    """Add entity nodes (places, people, organizations) - IMPROVED"""
    print("\n[Phase 3] Adding entity nodes...")
    
    all_locations = []
    all_persons = []
    all_orgs = []
    
    # Collect all entities
    for idx, doc in enumerate(documents):
        entities = doc.get('entities', {})
        
        # Handle both list and dict formats
        if isinstance(entities, dict):
            locations = entities.get('locations', [])
            persons = entities.get('persons', [])
            orgs = entities.get('organizations', [])
        else:
            locations = []
            persons = []
            orgs = []
        
        # Ensure they're lists
        if not isinstance(locations, list):
            locations = []
        if not isinstance(persons, list):
            persons = []
        if not isinstance(orgs, list):
            orgs = []
        
        all_locations.extend(locations)
        all_persons.extend(persons)
        all_orgs.extend(orgs)
        
        # Link document to its entities (top N only)
        for loc in locations[:CONFIG['max_entities_per_doc']]:
            if loc and isinstance(loc, str) and len(loc) > 2:
                loc_id = f"loc_{loc.lower().replace(' ', '_')[:50]}"
                if not G.has_node(loc_id):
                    G.add_node(loc_id, node_type='location', name=loc)
                G.add_edge(f"doc_{idx}", loc_id, relation='mentions_location', weight=1.0)
        
        for person in persons[:3]:
            if person and isinstance(person, str) and len(person) > 2:
                person_id = f"person_{person.lower().replace(' ', '_')[:50]}"
                if not G.has_node(person_id):
                    G.add_node(person_id, node_type='person', name=person)
                G.add_edge(f"doc_{idx}", person_id, relation='mentions_person', weight=1.0)
        
        for org in orgs[:3]:
            if org and isinstance(org, str) and len(org) > 2:
                org_id = f"org_{org.lower().replace(' ', '_')[:50]}"
                if not G.has_node(org_id):
                    G.add_node(org_id, node_type='organization', name=org)
                G.add_edge(f"doc_{idx}", org_id, relation='mentions_org', weight=1.0)
    
    location_counts = Counter(all_locations)
    person_counts = Counter(all_persons)
    org_counts = Counter(all_orgs)
    
    loc_nodes = len([n for n, d in G.nodes(data=True) if d.get('node_type') == 'location'])
    person_nodes = len([n for n, d in G.nodes(data=True) if d.get('node_type') == 'person'])
    org_nodes = len([n for n, d in G.nodes(data=True) if d.get('node_type') == 'organization'])
    
    print(f"  ✓ Added {loc_nodes} location nodes")
    print(f"  ✓ Added {person_nodes} person nodes")
    print(f"  ✓ Added {org_nodes} organization nodes")
    
    if location_counts:
        print(f"\n  Top locations: {location_counts.most_common(5)}")
    if person_counts:
        print(f"  Top persons: {person_counts.most_common(3)}")
    
    return loc_nodes, person_nodes, org_nodes

def add_concept_nodes(G, documents):
    """Add concept nodes (heritage types, domains, time periods)"""
    print("\n[Phase 4] Adding concept nodes...")
    
    all_heritage_types = set()
    all_domains = set()
    all_periods = set()
    all_regions = set()
    
    for idx, doc in enumerate(documents):
        classifications = doc['classifications']
        
        # Heritage types
        for htype in classifications.get('heritage_types', []):
            if htype:
                all_heritage_types.add(htype)
                htype_id = f"type_{htype}"
                if not G.has_node(htype_id):
                    G.add_node(htype_id, node_type='heritage_type', name=htype)
                G.add_edge(f"doc_{idx}", htype_id, relation='has_type', weight=1.0)
        
        # Domains
        for domain in classifications.get('domains', []):
            if domain:
                all_domains.add(domain)
                domain_id = f"domain_{domain}"
                if not G.has_node(domain_id):
                    G.add_node(domain_id, node_type='domain', name=domain)
                G.add_edge(f"doc_{idx}", domain_id, relation='belongs_to_domain', weight=1.0)
        
        # Time period
        period = classifications.get('time_period')
        if period and period != 'unknown':
            all_periods.add(period)
            period_id = f"period_{period}"
            if not G.has_node(period_id):
                G.add_node(period_id, node_type='time_period', name=period)
            G.add_edge(f"doc_{idx}", period_id, relation='from_period', weight=1.0)
        
        # Region
        region = classifications.get('region')
        if region and region != 'unknown':
            all_regions.add(region)
            region_id = f"region_{region}"
            if not G.has_node(region_id):
                G.add_node(region_id, node_type='region', name=region)
            G.add_edge(f"doc_{idx}", region_id, relation='located_in_region', weight=1.0)
    
    print(f"  ✓ Heritage types: {len(all_heritage_types)}")
    print(f"  ✓ Domains: {len(all_domains)}")
    print(f"  ✓ Time periods: {len(all_periods)}")
    print(f"  ✓ Regions: {len(all_regions)}")

def add_similarity_edges(G, documents, embeddings, threshold=0.6):
    """Add edges between similar documents based on cosine similarity"""
    print(f"\n[Phase 5] Computing document similarities (threshold={threshold})...")
    
    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    edges_added = 0
    
    for i in range(len(documents)):
        for j in range(i+1, len(documents)):
            sim = similarity_matrix[i][j]
            
            if sim >= threshold:
                G.add_edge(
                    f"doc_{i}",
                    f"doc_{j}",
                    relation='similar_to',
                    weight=float(sim),
                    similarity_type='embedding'
                )
                edges_added += 1
    
    print(f"  ✓ Added {edges_added} similarity edges (threshold {threshold})")

def add_concept_similarity_edges(G):
    """Add edges between similar concepts using Lesk similarity - IMPROVED"""
    print(f"\n[Phase 6] Computing concept similarities (Lesk, threshold={CONFIG['concept_similarity_threshold']})...")
    
    # Get all concept nodes
    heritage_types = [(n, d['name']) for n, d in G.nodes(data=True) if d.get('node_type') == 'heritage_type']
    domains = [(n, d['name']) for n, d in G.nodes(data=True) if d.get('node_type') == 'domain']
    periods = [(n, d['name']) for n, d in G.nodes(data=True) if d.get('node_type') == 'time_period']
    regions = [(n, d['name']) for n, d in G.nodes(data=True) if d.get('node_type') == 'region']
    
    edges_added = 0
    
    # Connect similar heritage types
    for i, (id1, name1) in enumerate(heritage_types):
        for id2, name2 in heritage_types[i+1:]:
            sim = compute_concept_similarity(name1, name2)
            if sim > CONFIG['concept_similarity_threshold']:
                G.add_edge(id1, id2, relation='semantically_related', weight=float(sim))
                edges_added += 1
    
    # Connect similar domains
    for i, (id1, name1) in enumerate(domains):
        for id2, name2 in domains[i+1:]:
            sim = compute_concept_similarity(name1, name2)
            if sim > CONFIG['concept_similarity_threshold']:
                G.add_edge(id1, id2, relation='semantically_related', weight=float(sim))
                edges_added += 1
    
    # Connect similar time periods
    for i, (id1, name1) in enumerate(periods):
        for id2, name2 in periods[i+1:]:
            sim = compute_concept_similarity(name1, name2)
            if sim > CONFIG['concept_similarity_threshold']:
                G.add_edge(id1, id2, relation='temporally_related', weight=float(sim))
                edges_added += 1
    
    # Connect similar regions
    for i, (id1, name1) in enumerate(regions):
        for id2, name2 in regions[i+1:]:
            sim = compute_concept_similarity(name1, name2)
            if sim > CONFIG['concept_similarity_threshold']:
                G.add_edge(id1, id2, relation='geographically_related', weight=float(sim))
                edges_added += 1
    
    print(f"  ✓ Added {edges_added} concept similarity edges")
    
    return edges_added

def add_cluster_edges(G, documents, embeddings):
    """
    Add edges connecting documents in the same cluster - IMPROVED
    Only connect to top-K most similar documents within cluster
    """
    print(f"\n[Phase 7] Adding cluster relationships (top-{CONFIG['top_k_cluster_connections']} per cluster)...")
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Group documents by cluster
    clusters = defaultdict(list)
    
    for idx, doc in enumerate(documents):
        clusters[doc['cluster_id']].append(idx)
    
    edges_added = 0
    
    for cluster_id, doc_indices in clusters.items():
        # For each document in the cluster
        for doc_idx in doc_indices:
            # Get similarities to other docs in same cluster
            similarities = []
            for other_idx in doc_indices:
                if other_idx != doc_idx:
                    sim = similarity_matrix[doc_idx][other_idx]
                    similarities.append((other_idx, sim))
            
            # Sort by similarity and keep top-K
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k = similarities[:CONFIG['top_k_cluster_connections']]
            
            # Add edges to top-K similar documents
            for other_idx, sim in top_k:
                doc_id_1 = f"doc_{doc_idx}"
                doc_id_2 = f"doc_{other_idx}"
                
                # Only add if edge doesn't exist (avoid duplicates)
                if not G.has_edge(doc_id_1, doc_id_2):
                    G.add_edge(
                        doc_id_1,
                        doc_id_2,
                        relation='same_cluster',
                        weight=float(sim),
                        cluster_id=cluster_id
                    )
                    edges_added += 1
    
    print(f"  ✓ Added {edges_added} cluster edges (reduced from {sum(len(docs)*(len(docs)-1)//2 for docs in clusters.values())})")

def compute_graph_statistics(G):
    """Compute and save graph statistics"""
    print("\n[Phase 8] Computing graph statistics...")
    
    stats = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'node_types': {},
        'edge_types': {},
        'density': nx.density(G),
        'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'is_connected': nx.is_connected(G),
        'number_of_components': nx.number_connected_components(G),
        'creation_date': datetime.now().isoformat(),
        'config': CONFIG
    }
    
    # Count node types
    for node, data in G.nodes(data=True):
        node_type = data.get('node_type', 'unknown')
        stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
    
    # Count edge types
    for u, v, data in G.edges(data=True):
        relation = data.get('relation', 'unknown')
        stats['edge_types'][relation] = stats['edge_types'].get(relation, 0) + 1
    
    # Compute centrality for document nodes
    doc_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'document']
    
    if len(doc_nodes) > 0:
        degree_cent = nx.degree_centrality(G)
        top_central_docs = sorted(
            [(n, degree_cent[n]) for n in doc_nodes],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        stats['top_central_documents'] = [
            {
                'title': G.nodes[n]['title'],
                'centrality': cent
            }
            for n, cent in top_central_docs
        ]
    
    print(f"\n  📊 Graph Statistics:")
    print(f"     Nodes: {stats['total_nodes']}")
    print(f"     Edges: {stats['total_edges']}")
    print(f"     Density: {stats['density']:.4f}")
    print(f"     Avg Degree: {stats['average_degree']:.2f}")
    print(f"     Connected: {stats['is_connected']}")
    print(f"     Components: {stats['number_of_components']}")
    
    print(f"\n  📈 Node Distribution:")
    for node_type, count in stats['node_types'].items():
        print(f"     {node_type}: {count}")
    
    print(f"\n  🔗 Edge Distribution:")
    for relation, count in stats['edge_types'].items():
        print(f"     {relation}: {count}")
    
    return stats

def visualize_graph_sample(G, documents):
    """Visualize a sample of the graph"""
    print("\n[Phase 9] Creating visualization...")
    
    # Sample: First 30 documents + their connected nodes
    sample_doc_nodes = [f"doc_{i}" for i in range(min(30, len(documents)))]
    
    # Get all neighbors of sample documents
    sample_nodes = set(sample_doc_nodes)
    for node in sample_doc_nodes:
        sample_nodes.update(G.neighbors(node))
    
    # Create subgraph
    subgraph = G.subgraph(sample_nodes)
    
    # Create layout
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
    
    # Color nodes by type
    node_colors = []
    for node in subgraph.nodes():
        node_type = subgraph.nodes[node].get('node_type', 'unknown')
        color_map = {
            'document': '#3498db',  # Blue
            'location': '#e74c3c',  # Red
            'person': '#2ecc71',  # Green
            'organization': '#f39c12',  # Orange
            'heritage_type': '#9b59b6',  # Purple
            'domain': '#1abc9c',  # Teal
            'time_period': '#e67e22',  # Dark orange
            'region': '#34495e'  # Dark gray
        }
        node_colors.append(color_map.get(node_type, '#95a5a6'))
    
    # Draw
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=300, alpha=0.7)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5)
    
    # Labels for document nodes only
    doc_labels = {n: subgraph.nodes[n].get('title', '')[:20] + '...' 
                  for n in subgraph.nodes() if subgraph.nodes[n].get('node_type') == 'document'}
    nx.draw_networkx_labels(subgraph, pos, doc_labels, font_size=6)
    
    plt.title('Heritage Knowledge Graph (Sample) - IMPROVED', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(KG_VIZ_FILE, dpi=300, bbox_inches='tight')
    print(f"  ✓ Visualization saved to: {KG_VIZ_FILE}")
    plt.close()

def save_graph(G, stats):
    """Save the knowledge graph"""
    print("\n[Phase 10] Saving knowledge graph...")
    
    os.makedirs(KG_DIR, exist_ok=True)
    
    # Save graph as pickle (preserves all attributes)
    with open(KG_FILE, "wb") as f:
        pickle.dump(G, f)
    print(f"  ✓ Graph saved to: {KG_FILE}")
    
    # Save statistics
    with open(KG_STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Statistics saved to: {KG_STATS_FILE}")
    
    # Also save in GML format (human-readable)
    gml_file = os.path.join(KG_DIR, "heritage_kg.gml")
    nx.write_gml(G, gml_file)
    print(f"  ✓ GML format saved to: {gml_file}")

def main():
    print("="*70)
    print("KNOWLEDGE GRAPH CONSTRUCTION (IMPROVED)")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n⚙️  Configuration:")
    print(f"   - Similarity threshold: {CONFIG['similarity_threshold']}")
    print(f"   - Concept similarity threshold: {CONFIG['concept_similarity_threshold']}")
    print(f"   - Top-K cluster connections: {CONFIG['top_k_cluster_connections']}")
    
    # Load data
    documents, embeddings = load_data()
    
    # Create base graph
    G = create_base_graph(documents)
    
    # Add entity nodes
    loc_count, person_count, org_count = add_entity_nodes(G, documents)
    
    # Add concept nodes
    add_concept_nodes(G, documents)
    
    # Add similarity edges
    add_similarity_edges(G, documents, embeddings, threshold=CONFIG['similarity_threshold'])
    
    # Add concept similarity edges (Lesk) - IMPROVED
    concept_edges = add_concept_similarity_edges(G)
    
    # Add cluster edges - IMPROVED (top-K only)
    add_cluster_edges(G, documents, embeddings)
    
    # Compute statistics
    stats = compute_graph_statistics(G)
    
    # Visualize
    visualize_graph_sample(G, documents)
    
    # Save
    save_graph(G, stats)
    
    # Summary
    print("\n" + "="*70)
    print("✅ KNOWLEDGE GRAPH CONSTRUCTION COMPLETE")
    print("="*70)
    print(f"Built OPTIMIZED KG with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"\n📊 Key improvements:")
    print(f"   - Reduced cluster edges from ~6,500 to ~{stats['edge_types'].get('same_cluster', 0)}")
    print(f"   - Added {concept_edges} concept similarity edges")
    print(f"   - Entity nodes: {loc_count} locations, {person_count} persons, {org_count} orgs")
    print(f"\n📁 Files created:")
    print(f"   - {KG_FILE}")
    print(f"   - {KG_STATS_FILE}")
    print(f"   - {KG_VIZ_FILE}")
    print("="*70)

if __name__ == "__main__":
    main()