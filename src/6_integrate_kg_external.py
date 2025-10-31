import json
import os
import networkx as nx
import pickle
from collections import defaultdict, Counter
from datetime import datetime
import matplotlib.pyplot as plt
import requests
import time
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib.parse

# Directories
KG_DIR = "knowledge_graph"
CLASSIFIED_DIR = "data/classified"

# Files
KG_FILE = os.path.join(KG_DIR, "heritage_kg.gpickle")
INTEGRATED_KG_FILE = os.path.join(KG_DIR, "heritage_kg_integrated.gpickle")
SUBGRAPHS_DIR = os.path.join(KG_DIR, "subgraphs")
INTEGRATION_STATS_FILE = os.path.join(KG_DIR, "integration_statistics.json")
ENRICHMENT_LOG_FILE = os.path.join(KG_DIR, "external_enrichment_log.json")

# ========== DBPEDIA INTEGRATION ==========

def query_dbpedia(entity_name):
    """Query DBpedia SPARQL endpoint for entity information"""
    try:
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        
        # Clean entity name for query
        clean_name = entity_name.replace("'", "\\'")
        
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
        
        SELECT DISTINCT ?resource ?abstract ?lat ?long ?type ?country WHERE {{
            ?resource rdfs:label "{clean_name}"@en .
            OPTIONAL {{ ?resource dbo:abstract ?abstract . FILTER(LANG(?abstract) = "en") }}
            OPTIONAL {{ ?resource geo:lat ?lat }}
            OPTIONAL {{ ?resource geo:long ?long }}
            OPTIONAL {{ ?resource rdf:type ?type }}
            OPTIONAL {{ ?resource dbo:country ?country }}
        }} LIMIT 5
        """
        
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(15)
        
        results = sparql.query().convert()
        
        if results['results']['bindings']:
            result = results['results']['bindings'][0]
            
            enrichment = {
                'dbpedia_uri': result.get('resource', {}).get('value', ''),
                'abstract': result.get('abstract', {}).get('value', '')[:500],
                'coordinates': {
                    'lat': float(result.get('lat', {}).get('value', 0)) if result.get('lat') else None,
                    'long': float(result.get('long', {}).get('value', 0)) if result.get('long') else None
                },
                'type': result.get('type', {}).get('value', ''),
                'country': result.get('country', {}).get('value', ''),
                'source': 'DBpedia'
            }
            
            return enrichment
        
        return None
        
    except Exception as e:
        return None

def query_wikidata(entity_name, retries=2):
    """Query Wikidata API for entity information with retry logic"""
    for attempt in range(retries):
        try:
            # Search for entity
            search_url = "https://www.wikidata.org/w/api.php"
            search_params = {
                'action': 'wbsearchentities',
                'format': 'json',
                'language': 'en',
                'search': entity_name,
                'limit': 1
            }
            
            response = requests.get(search_url, params=search_params, timeout=15)
            
            # Check if response is valid JSON
            if response.status_code != 200:
                if attempt < retries - 1:
                    time.sleep(3)  # Wait longer before retry
                    continue
                return None
            
            search_results = response.json()
            
            if not search_results.get('search'):
                return None
            
            entity_id = search_results['search'][0]['id']
            
            # Get entity details
            entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
            entity_response = requests.get(entity_url, timeout=15)
            
            if entity_response.status_code != 200:
                if attempt < retries - 1:
                    time.sleep(3)
                    continue
                return None
            
            entity_data = entity_response.json()
            
            entity_info = entity_data['entities'][entity_id]
            
            enrichment = {
                'wikidata_id': entity_id,
                'wikidata_uri': f"https://www.wikidata.org/entity/{entity_id}",
                'description': entity_info.get('descriptions', {}).get('en', {}).get('value', ''),
                'aliases': [alias['value'] for alias in entity_info.get('aliases', {}).get('en', [])[:3]],
                'source': 'Wikidata'
            }
            
            # Extract specific properties
            claims = entity_info.get('claims', {})
            
            # Coordinates (P625)
            if 'P625' in claims:
                try:
                    coords = claims['P625'][0]['mainsnak']['datavalue']['value']
                    enrichment['coordinates'] = {
                        'lat': coords.get('latitude'),
                        'long': coords.get('longitude')
                    }
                except:
                    pass
            
            # Instance of (P31)
            if 'P31' in claims:
                try:
                    enrichment['instance_of'] = []
                    for claim in claims['P31'][:3]:
                        if 'datavalue' in claim['mainsnak']:
                            enrichment['instance_of'].append(claim['mainsnak']['datavalue']['value']['id'])
                except:
                    pass
            
            # Country (P17)
            if 'P17' in claims:
                try:
                    country_id = claims['P17'][0]['mainsnak']['datavalue']['value']['id']
                    enrichment['country_id'] = country_id
                except:
                    pass
            
            # Inception/Founded (P571)
            if 'P571' in claims:
                try:
                    date_value = claims['P571'][0]['mainsnak']['datavalue']['value']
                    enrichment['inception_date'] = date_value.get('time', '')
                except:
                    pass
            
            # UNESCO World Heritage Site (P1435)
            if 'P1435' in claims:
                enrichment['unesco_status'] = True
            
            return enrichment
            
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(3)
                continue
            return None
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                time.sleep(3)
                continue
            return None
        except (ValueError, KeyError, json.JSONDecodeError):
            # JSON parsing error or missing keys
            if attempt < retries - 1:
                time.sleep(3)
                continue
            return None
        except Exception:
            return None
    
    return None

# ========== ENRICHMENT FUNCTIONS ==========

def load_knowledge_graph():
    """Load the constructed knowledge graph"""
    print("\n[Phase 1] Loading knowledge graph...")
    
    with open(KG_FILE, 'rb') as f:
        G = pickle.load(f)
    
    print(f"✓ Loaded KG with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G

def enrich_location_entities(G):
    """Enrich location entities with DBpedia and Wikidata"""
    print("\n[Phase 2] Enriching location entities from DBpedia & Wikidata...")
    print("  Note: Wikidata API may be slow or fail - we'll rely more on DBpedia")
    
    location_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('node_type') == 'location']
    
    # Sort by degree (most connected first)
    location_nodes = sorted(location_nodes, key=lambda x: G.degree(x[0]), reverse=True)
    
    # Enrich top 50 locations
    enriched_count = 0
    enrichment_log = []
    wikidata_failures = 0
    
    for node_id, node_data in location_nodes[:50]:
        location_name = node_data.get('name', '')
        print(f"  [{enriched_count+1}/50] Enriching: {location_name}")
        
        # Try DBpedia first (more reliable)
        dbpedia_data = query_dbpedia(location_name)
        time.sleep(1)  # Rate limiting
        
        # Try Wikidata only if we haven't had too many failures
        wikidata_data = None
        if wikidata_failures < 10:  # Stop trying after 10 consecutive failures
            wikidata_data = query_wikidata(location_name)
            if wikidata_data is None:
                wikidata_failures += 1
            else:
                wikidata_failures = 0  # Reset counter on success
            time.sleep(2)  # Longer wait for Wikidata
        
        # Merge enrichments
        if dbpedia_data or wikidata_data:
            node_data['external_knowledge'] = {}
            
            if dbpedia_data:
                node_data['external_knowledge']['dbpedia'] = dbpedia_data
                print(f"    ✓ DBpedia: {dbpedia_data.get('abstract', '')[:60]}...")
            
            if wikidata_data:
                node_data['external_knowledge']['wikidata'] = wikidata_data
                print(f"    ✓ Wikidata: {wikidata_data.get('description', '')[:60]}")
            
            # Set coordinates if available
            coords = None
            if dbpedia_data and dbpedia_data.get('coordinates', {}).get('lat'):
                coords = dbpedia_data['coordinates']
            elif wikidata_data and wikidata_data.get('coordinates'):
                coords = wikidata_data['coordinates']
            
            if coords and coords.get('lat'):
                node_data['coordinates'] = coords
            
            enriched_count += 1
            enrichment_log.append({
                'entity': location_name,
                'type': 'location',
                'dbpedia': dbpedia_data is not None,
                'wikidata': wikidata_data is not None
            })
    
    print(f"\n  ✓ Enriched {enriched_count} locations")
    print(f"  📊 DBpedia: {sum(1 for e in enrichment_log if e['dbpedia'])}")
    print(f"  📊 Wikidata: {sum(1 for e in enrichment_log if e['wikidata'])}")
    
    return enriched_count, enrichment_log

def enrich_person_entities(G):
    """Enrich person entities with DBpedia and Wikidata"""
    print("\n[Phase 3] Enriching person entities from DBpedia & Wikidata...")
    
    person_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('node_type') == 'person']
    
    # Sort by degree
    person_nodes = sorted(person_nodes, key=lambda x: G.degree(x[0]), reverse=True)
    
    enriched_count = 0
    enrichment_log = []
    wikidata_failures = 0
    
    for node_id, node_data in person_nodes[:30]:  # Top 30 persons
        person_name = node_data.get('name', '')
        print(f"  [{enriched_count+1}/30] Enriching: {person_name}")
        
        # Try DBpedia
        dbpedia_data = query_dbpedia(person_name)
        time.sleep(1)
        
        # Try Wikidata only if not too many failures
        wikidata_data = None
        if wikidata_failures < 10:
            wikidata_data = query_wikidata(person_name)
            if wikidata_data is None:
                wikidata_failures += 1
            else:
                wikidata_failures = 0
            time.sleep(2)
        
        if dbpedia_data or wikidata_data:
            node_data['external_knowledge'] = {}
            
            if dbpedia_data:
                node_data['external_knowledge']['dbpedia'] = dbpedia_data
                print(f"    ✓ DBpedia enriched")
            
            if wikidata_data:
                node_data['external_knowledge']['wikidata'] = wikidata_data
                print(f"    ✓ Wikidata enriched")
            
            enriched_count += 1
            enrichment_log.append({
                'entity': person_name,
                'type': 'person',
                'dbpedia': dbpedia_data is not None,
                'wikidata': wikidata_data is not None
            })
    
    print(f"\n  ✓ Enriched {enriched_count} persons")
    print(f"  📊 DBpedia: {sum(1 for e in enrichment_log if e['dbpedia'])}")
    print(f"  📊 Wikidata: {sum(1 for e in enrichment_log if e['wikidata'])}")
    
    return enriched_count, enrichment_log

def enrich_organization_entities(G):
    """Enrich organization entities"""
    print("\n[Phase 4] Enriching organization entities...")
    print("  (Focusing on DBpedia due to Wikidata API issues)")
    
    org_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('node_type') == 'organization']
    org_nodes = sorted(org_nodes, key=lambda x: G.degree(x[0]), reverse=True)
    
    enriched_count = 0
    enrichment_log = []
    
    for node_id, node_data in org_nodes[:20]:  # Top 20 orgs
        org_name = node_data.get('name', '')
        print(f"  [{enriched_count+1}/20] Enriching: {org_name}")
        
        # Focus on DBpedia for organizations
        dbpedia_data = query_dbpedia(org_name)
        time.sleep(1)
        
        if dbpedia_data:
            node_data['external_knowledge'] = {'dbpedia': dbpedia_data}
            print(f"    ✓ DBpedia: {dbpedia_data.get('abstract', '')[:60]}...")
            
            enriched_count += 1
            enrichment_log.append({
                'entity': org_name,
                'type': 'organization',
                'dbpedia': True,
                'wikidata': False
            })
    
    print(f"\n  ✓ Enriched {enriched_count} organizations with DBpedia")
    
    return enriched_count, enrichment_log

def add_geographic_relationships(G):
    """Add geographic hierarchy from external knowledge"""
    print("\n[Phase 5] Adding geographic relationships from external data...")
    
    edges_added = 0
    
    # Indian geographic hierarchy
    geographic_hierarchy = {
        'India': ['Delhi', 'Agra', 'Gujarat', 'Rajasthan', 'Tamil Nadu', 'Karnataka', 
                  'Maharashtra', 'Uttar Pradesh', 'Madhya Pradesh', 'West Bengal', 'Kerala'],
        'Delhi': ['Red Fort', 'Qutub Minar', 'India Gate'],
        'Agra': ['Taj Mahal', 'Agra Fort', 'Fatehpur Sikri'],
        'Rajasthan': ['Jaipur', 'Udaipur', 'Jaisalmer', 'Jodhpur'],
        'Gujarat': ['Ahmedabad', 'Dholavira'],
    }
    
    for parent, children in geographic_hierarchy.items():
        parent_id = f"loc_{parent.lower().replace(' ', '_')}"
        
        # Ensure parent exists
        if not G.has_node(parent_id):
            G.add_node(parent_id, node_type='location', name=parent)
        
        # Add hierarchy edges
        for child in children:
            child_id = f"loc_{child.lower().replace(' ', '_')}"
            
            # Check if child exists in graph
            if G.has_node(child_id):
                if not G.has_edge(parent_id, child_id):
                    G.add_edge(parent_id, child_id, relation='contains', weight=0.9)
                    edges_added += 1
    
    # Connect locations based on coordinates (nearby locations)
    location_nodes = [(n, d) for n, d in G.nodes(data=True) 
                      if d.get('node_type') == 'location' and d.get('coordinates')]
    
    # Simple distance-based clustering
    for i, (node1, data1) in enumerate(location_nodes):
        coords1 = data1['coordinates']
        if not coords1.get('lat') or not coords1.get('long'):
            continue
        
        for node2, data2 in location_nodes[i+1:]:
            coords2 = data2['coordinates']
            if not coords2.get('lat') or not coords2.get('long'):
                continue
            
            # Calculate simple Euclidean distance
            lat_diff = coords1['lat'] - coords2['lat']
            long_diff = coords1['long'] - coords2['long']
            distance = (lat_diff**2 + long_diff**2)**0.5
            
            # If within ~200km (rough approximation)
            if distance < 2.0:
                if not G.has_edge(node1, node2):
                    G.add_edge(node1, node2, relation='geographically_nearby', 
                             weight=0.7, distance_approx=distance)
                    edges_added += 1
    
    print(f"  ✓ Added {edges_added} geographic relationship edges")
    
    return edges_added

def add_temporal_relationships(G):
    """Add temporal relationships"""
    print("\n[Phase 6] Adding temporal relationships...")
    
    time_sequence = ['ancient', 'medieval', 'modern']
    period_nodes = [f"period_{p}" for p in time_sequence if G.has_node(f"period_{p}")]
    
    edges_added = 0
    for i in range(len(period_nodes) - 1):
        if not G.has_edge(period_nodes[i], period_nodes[i+1]):
            G.add_edge(period_nodes[i], period_nodes[i+1], 
                      relation='temporally_follows', weight=0.9)
            edges_added += 1
    
    print(f"  ✓ Added {edges_added} temporal sequence edges")
    
    return edges_added

def add_domain_relationships(G):
    """Connect related domains"""
    print("\n[Phase 7] Adding domain relationships...")
    
    # Domain relationships (from domain knowledge)
    domain_relations = [
        ('domain_religious', 'domain_cultural', 0.8),
        ('domain_religious', 'domain_architectural', 0.7),
        ('domain_military', 'domain_royal', 0.7),
        ('domain_archaeological', 'domain_architectural', 0.8),
    ]
    
    edges_added = 0
    
    for domain1, domain2, weight in domain_relations:
        if G.has_node(domain1) and G.has_node(domain2):
            if not G.has_edge(domain1, domain2):
                G.add_edge(domain1, domain2, relation='related_domain', weight=weight)
                edges_added += 1
    
    print(f"  ✓ Added {edges_added} domain relationship edges")
    
    return edges_added

def add_cross_entity_relationships(G):
    """Add relationships between different entity types"""
    print("\n[Phase 8] Adding cross-entity relationships...")
    
    # Connect locations to regions
    location_to_region = {
        'loc_delhi': 'region_north',
        'loc_agra': 'region_north',
        'loc_jaipur': 'region_north',
        'loc_gujarat': 'region_west',
        'loc_mumbai': 'region_west',
        'loc_chennai': 'region_south',
        'loc_bangalore': 'region_south',
        'loc_kolkata': 'region_east',
    }
    
    edges_added = 0
    
    for loc_id, region_id in location_to_region.items():
        if G.has_node(loc_id) and G.has_node(region_id):
            if not G.has_edge(loc_id, region_id):
                G.add_edge(loc_id, region_id, relation='geographically_related', weight=0.9)
                edges_added += 1
    
    # Connect prominent persons to locations (simplified)
    person_locations = [
        ('person_akbar', 'loc_agra', 0.9),
        ('person_shah_jahan', 'loc_agra', 0.95),
        ('person_ashoka', 'loc_india', 0.9),
    ]
    
    for person_id, loc_id, weight in person_locations:
        if G.has_node(person_id) and G.has_node(loc_id):
            if not G.has_edge(person_id, loc_id):
                G.add_edge(person_id, loc_id, relation='associated_with', weight=weight)
                edges_added += 1
    
    print(f"  ✓ Added {edges_added} cross-entity relationship edges")
    
    return edges_added

def compute_centrality_metrics(G):
    """Compute centrality metrics (needed for ranking)"""
    print("\n[Phase 9] Computing centrality metrics...")
    
    # PageRank (important for SimRank later)
    print("  Computing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
    
    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    
    # Store as node attributes
    for node in G.nodes():
        G.nodes[node]['degree_centrality'] = degree_cent.get(node, 0.0)
        G.nodes[node]['pagerank'] = pagerank.get(node, 0.0)
    
    # Top nodes by PageRank
    top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:15]
    
    print(f"\n  📊 Top 15 nodes by PageRank:")
    for node, score in top_pagerank:
        node_data = G.nodes[node]
        node_type = node_data.get('node_type', 'unknown')
        name = node_data.get('title', node_data.get('name', node))
        print(f"     {name[:50]} ({node_type}): {score:.6f}")
    
    print(f"\n  ✓ Computed centrality metrics for all {G.number_of_nodes()} nodes")
    
    return {
        'top_pagerank': [(G.nodes[n].get('title', G.nodes[n].get('name', n)), score) 
                         for n, score in top_pagerank]
    }

def create_domain_subgraphs(G):
    """Create subgraphs for each domain"""
    print("\n[Phase 10] Creating domain subgraphs...")
    
    os.makedirs(SUBGRAPHS_DIR, exist_ok=True)
    
    domain_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'domain']
    
    subgraph_info = {}
    
    for domain_node in domain_nodes:
        domain_name = G.nodes[domain_node]['name']
        
        # Get all documents connected to this domain
        connected_docs = [neighbor for neighbor in G.neighbors(domain_node)
                         if G.nodes[neighbor].get('node_type') == 'document']
        
        if not connected_docs:
            continue
        
        # Create subgraph
        subgraph_nodes = set([domain_node] + connected_docs)
        for doc in connected_docs:
            subgraph_nodes.update(G.neighbors(doc))
        
        subgraph = G.subgraph(subgraph_nodes)
        
        # Save subgraph
        subgraph_file = os.path.join(SUBGRAPHS_DIR, f"subgraph_{domain_name}.gpickle")
        with open(subgraph_file, 'wb') as f:
            pickle.dump(subgraph, f)
        
        subgraph_info[domain_name] = {
            'nodes': subgraph.number_of_nodes(),
            'edges': subgraph.number_of_edges(),
            'documents': len(connected_docs),
            'file': subgraph_file
        }
        
        print(f"  ✓ {domain_name}: {len(connected_docs)} docs, "
              f"{subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    
    print(f"\n  ✓ Created {len(subgraph_info)} domain subgraphs")
    
    return subgraph_info

def visualize_enriched_graph(G):
    """Visualize enriched entities"""
    print("\n[Phase 11] Creating enriched graph visualization...")
    
    # Find enriched nodes
    enriched_locations = [n for n, d in G.nodes(data=True) 
                          if d.get('node_type') == 'location' 
                          and 'external_knowledge' in d][:10]
    
    # Get their neighborhoods
    viz_nodes = set(enriched_locations)
    for loc in enriched_locations:
        viz_nodes.update(list(G.neighbors(loc))[:5])
    
    subgraph = G.subgraph(viz_nodes)
    
    # Visualize
    plt.figure(figsize=(18, 12))
    pos = nx.spring_layout(subgraph, k=3, iterations=50, seed=42)
    
    # Color enriched nodes differently
    node_colors = []
    for node in subgraph.nodes():
        has_external = 'external_knowledge' in subgraph.nodes[node]
        node_type = subgraph.nodes[node].get('node_type', 'unknown')
        
        if has_external:
            node_colors.append('#e74c3c')  # Red for enriched
        else:
            color_map = {
                'document': '#3498db',
                'location': '#f39c12',
                'person': '#2ecc71',
                'organization': '#9b59b6'
            }
            node_colors.append(color_map.get(node_type, '#95a5a6'))
    
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                          node_size=300, alpha=0.7)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=1.0)
    
    # Label enriched nodes
    labels = {n: subgraph.nodes[n].get('name', '')[:15] 
              for n in subgraph.nodes() 
              if 'external_knowledge' in subgraph.nodes[n]}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
    
    plt.title('Knowledge Graph with External Enrichment\n(Red = DBpedia/Wikidata enriched)', 
             fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    viz_file = os.path.join(KG_DIR, 'kg_external_enriched.png')
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Visualization saved to: {viz_file}")
    plt.close()

def save_integrated_graph(G, stats, enrichment_log):
    """Save integrated graph and statistics"""
    print("\n[Phase 12] Saving integrated knowledge graph...")
    
    # Save graph
    with open(INTEGRATED_KG_FILE, 'wb') as f:
        pickle.dump(G, f)
    print(f"  ✓ Integrated graph saved to: {INTEGRATED_KG_FILE}")
    
    # Save statistics
    with open(INTEGRATION_STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Statistics saved to: {INTEGRATION_STATS_FILE}")
    
    # Save enrichment log
    with open(ENRICHMENT_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(enrichment_log, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Enrichment log saved to: {ENRICHMENT_LOG_FILE}")

def main():
    print("="*70)
    print("KNOWLEDGE GRAPH INTEGRATION WITH EXTERNAL KNOWLEDGE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nExternal Sources:")
    print("  - DBpedia SPARQL Endpoint")
    print("  - Wikidata API (with fallback to DBpedia only)")
    print("\nThis will take ~20-30 minutes due to API rate limiting...")
    print("="*70)
    
    # Load KG
    G = load_knowledge_graph()
    
    initial_nodes = G.number_of_nodes()
    initial_edges = G.number_of_edges()
    
    # External enrichment
    loc_enriched, loc_log = enrich_location_entities(G)
    person_enriched, person_log = enrich_person_entities(G)
    org_enriched, org_log = enrich_organization_entities(G)
    
    # Internal enrichment
    geo_edges = add_geographic_relationships(G)
    temp_edges = add_temporal_relationships(G)
    domain_edges = add_domain_relationships(G)
    cross_edges = add_cross_entity_relationships(G)
    
    # Centrality
    centrality_stats = compute_centrality_metrics(G)
    
    # Subgraphs
    subgraph_info = create_domain_subgraphs(G)
    
    # Visualize
    visualize_enriched_graph(G)
    
    # Statistics
    final_nodes = G.number_of_nodes()
    final_edges = G.number_of_edges()
    
    enrichment_log = loc_log + person_log + org_log
    
    stats = {
        'initial_nodes': initial_nodes,
        'initial_edges': initial_edges,
        'final_nodes': final_nodes,
        'final_edges': final_edges,
        'external_enrichment': {
          'locations_enriched': loc_enriched,
            'persons_enriched': person_enriched,
            'organizations_enriched': org_enriched,
            'total_enriched': loc_enriched + person_enriched + org_enriched,
            'dbpedia_success': sum(1 for e in enrichment_log if e.get('dbpedia')),
            'wikidata_success': sum(1 for e in enrichment_log if e.get('wikidata'))
        },
        'edges_added': {
            'geographic_hierarchy': geo_edges,
            'temporal_sequence': temp_edges,
            'domain_relationships': domain_edges,
            'cross_entity': cross_edges,
            'total': geo_edges + temp_edges + domain_edges + cross_edges
        },
        'centrality_metrics': centrality_stats,
        'subgraphs': subgraph_info,
        'enrichment_sources': ['DBpedia', 'Wikidata'],
        'integration_date': datetime.now().isoformat()
    }
    
    # Save
    save_integrated_graph(G, stats, enrichment_log)
    
    # Summary
    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH INTEGRATION COMPLETE")
    print("="*70)
    print(f"📈 Graph Growth:")
    print(f"   Nodes: {initial_nodes} → {final_nodes}")
    print(f"   Edges: {initial_edges} → {final_edges} (+{final_edges - initial_edges})")
    print(f"\n🌐 External Enrichment:")
    print(f"   Locations enriched: {loc_enriched}")
    print(f"   Persons enriched: {person_enriched}")
    print(f"   Organizations enriched: {org_enriched}")
    print(f"   Total entities enriched: {loc_enriched + person_enriched + org_enriched}")
    print(f"   DBpedia success: {stats['external_enrichment']['dbpedia_success']}")
    print(f"   Wikidata success: {stats['external_enrichment']['wikidata_success']}")
    print(f"\n🔗 New Relationships:")
    print(f"   Geographic: {geo_edges}")
    print(f"   Temporal: {temp_edges}")
    print(f"   Domain: {domain_edges}")
    print(f"   Cross-entity: {cross_edges}")
    print(f"   Total new edges: {geo_edges + temp_edges + domain_edges + cross_edges}")
    print(f"\n📊 Domain Subgraphs: {len(subgraph_info)}")
    print(f"\n✅ Integrated KG saved to: {INTEGRATED_KG_FILE}")
    print(f"✅ Enrichment log: {ENRICHMENT_LOG_FILE}")
    print(f"\n⏱ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()