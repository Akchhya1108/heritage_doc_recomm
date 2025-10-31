import json
import os
import pickle
import networkx as nx
from datetime import datetime
from collections import defaultdict
import re

# Directories
KG_DIR = "knowledge_graph"
ONTOLOGY_DIR = "data/ontologies"
CLASSIFIED_DIR = "data/classified"

# Files
INTEGRATED_KG_FILE = os.path.join(KG_DIR, "heritage_kg_integrated.gpickle")
ONTOLOGY_FILE = os.path.join(ONTOLOGY_DIR, "heritage_ontologies.json")
CLASSIFIED_FILE = os.path.join(CLASSIFIED_DIR, "classified_documents.json")

# ========== ONTOLOGY DEFINITIONS ==========

class HeritageOntology:
    """Heritage Domain Ontology"""
    
    def __init__(self):
        # Hierarchical ontology structure
        self.ontology = {
            'HeritageType': {
                'TangibleHeritage': {
                    'Monument': ['Temple', 'Fort', 'Palace', 'Tomb', 'Memorial'],
                    'Site': ['ArchaeologicalSite', 'HistoricalSite', 'CulturalLandscape'],
                    'Artifact': ['Sculpture', 'Painting', 'Manuscript', 'Architecture']
                },
                'IntangibleHeritage': {
                    'Tradition': ['Festival', 'Ritual', 'Custom'],
                    'Art': ['Music', 'Dance', 'Craft']
                }
            },
            'Domain': {
                'Religious': ['Temple', 'Mosque', 'Church', 'Monastery', 'Shrine'],
                'Military': ['Fort', 'Fortress', 'Citadel', 'DefenseStructure'],
                'Royal': ['Palace', 'Court', 'Dynasty'],
                'Cultural': ['Festival', 'Tradition', 'Art'],
                'Archaeological': ['Excavation', 'Ruins', 'AncientSite'],
                'Architectural': ['Building', 'Structure', 'Design']
            },
            'TimePeriod': {
                'Ancient': ['Prehistoric', 'IndusValley', 'Vedic', 'Maurya', 'Gupta'],
                'Medieval': ['Sultanate', 'Mughal', 'Vijayanagar', 'Chola'],
                'Modern': ['Colonial', 'British', 'Contemporary']
            },
            'ArchitecturalStyle': {
                'IndoIslamic': ['Mughal', 'Sultanate', 'Dome', 'Minaret'],
                'Dravidian': ['Gopuram', 'Vimana', 'SouthIndian'],
                'Nagara': ['Shikhara', 'NorthIndian'],
                'Buddhist': ['Stupa', 'Chaitya', 'Vihara'],
                'Colonial': ['British', 'Gothic', 'Victorian']
            },
            'GeographicRegion': {
                'India': {
                    'North': ['Delhi', 'Punjab', 'Rajasthan', 'UttarPradesh'],
                    'South': ['TamilNadu', 'Karnataka', 'Kerala', 'AndhraPradesh'],
                    'East': ['WestBengal', 'Odisha', 'Bihar'],
                    'West': ['Gujarat', 'Maharashtra', 'Goa'],
                    'Central': ['MadhyaPradesh', 'Chhattisgarh']
                }
            }
        }
    
    def get_concept_hierarchy(self, concept):
        """Get hierarchical path for a concept"""
        def search_hierarchy(obj, target, path=[]):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.lower() == target.lower():
                        return path + [key]
                    result = search_hierarchy(value, target, path + [key])
                    if result:
                        return result
            elif isinstance(obj, list):
                if target.lower() in [item.lower() for item in obj]:
                    return path + [target]
            return None
        
        return search_hierarchy(self.ontology, concept)
    
    def get_related_concepts(self, concept, max_distance=2):
        """Get related concepts within ontology"""
        hierarchy = self.get_concept_hierarchy(concept)
        if not hierarchy:
            return []
        
        # Get siblings and cousins
        related = []
        
        # Navigate to parent and get siblings
        def get_siblings(obj, path, depth=0):
            if depth >= len(path):
                return []
            
            if isinstance(obj, dict):
                if depth == len(path) - 1:
                    # Found parent level, return siblings
                    parent_key = path[depth]
                    if parent_key in obj:
                        if isinstance(obj[parent_key], dict):
                            return list(obj[parent_key].keys())
                        elif isinstance(obj[parent_key], list):
                            return obj[parent_key]
                else:
                    # Keep navigating
                    for key, value in obj.items():
                        if key == path[depth]:
                            return get_siblings(value, path, depth + 1)
            return []
        
        related = get_siblings(self.ontology, hierarchy)
        
        return [r for r in related if r.lower() != concept.lower()]

# ========== DOCUMENT ONTOLOGY GENERATION ==========

def generate_document_ontology(doc, heritage_ontology):
    """Generate ontology representation for a document"""
    
    ontology = {
        'document_id': doc.get('file_name', ''),
        'title': doc.get('title', ''),
        'concepts': {},
        'properties': {},
        'relationships': []
    }
    
    # Extract concepts from classifications
    classifications = doc.get('classifications', {})
    
    # Heritage Types
    heritage_types = classifications.get('heritage_types', [])
    ontology['concepts']['HeritageType'] = heritage_types
    for htype in heritage_types:
        hierarchy = heritage_ontology.get_concept_hierarchy(htype)
        if hierarchy:
            ontology['concepts'][f'HeritageType_Hierarchy_{htype}'] = hierarchy
    
    # Domains
    domains = classifications.get('domains', [])
    ontology['concepts']['Domain'] = domains
    for domain in domains:
        hierarchy = heritage_ontology.get_concept_hierarchy(domain)
        if hierarchy:
            ontology['concepts'][f'Domain_Hierarchy_{domain}'] = hierarchy
    
    # Time Period
    time_period = classifications.get('time_period', '')
    if time_period and time_period != 'unknown':
        ontology['concepts']['TimePeriod'] = [time_period]
        hierarchy = heritage_ontology.get_concept_hierarchy(time_period)
        if hierarchy:
            ontology['concepts']['TimePeriod_Hierarchy'] = hierarchy
    
    # Architectural Style
    arch_styles = classifications.get('architectural_styles', [])
    if arch_styles:
        ontology['concepts']['ArchitecturalStyle'] = arch_styles
    
    # Region
    region = classifications.get('region', '')
    if region and region != 'unknown':
        ontology['concepts']['GeographicRegion'] = [region]
    
    # Properties
    ontology['properties']['tangibility'] = classifications.get('tangibility', 'tangible')
    ontology['properties']['source'] = doc.get('source', '')
    ontology['properties']['cluster_id'] = doc.get('cluster_id', -1)
    ontology['properties']['cluster_label'] = doc.get('cluster_label', '')
    
    # Extract entities
    entities = doc.get('entities', {})
    ontology['properties']['locations'] = entities.get('locations', [])[:5]
    ontology['properties']['persons'] = entities.get('persons', [])[:5]
    ontology['properties']['organizations'] = entities.get('organizations', [])[:3]
    
    # Relationships (semantic)
    for htype in heritage_types:
        related = heritage_ontology.get_related_concepts(htype)
        for rel in related:
            ontology['relationships'].append({
                'type': 'related_heritage_type',
                'from': htype,
                'to': rel,
                'weight': 0.7
            })
    
    return ontology

# ========== QUERY ONTOLOGY GENERATION ==========

def parse_query(query_text):
    """Parse user query and extract concepts"""
    query_text = query_text.lower()
    
    # Extract potential concepts
    concepts = {
        'heritage_types': [],
        'domains': [],
        'time_periods': [],
        'architectural_styles': [],
        'regions': [],
        'keywords': []
    }
    
    # Heritage type keywords
    heritage_keywords = {
        'temple': 'monument', 'fort': 'monument', 'palace': 'monument',
        'mosque': 'monument', 'church': 'monument', 'tomb': 'monument',
        'site': 'site', 'ruins': 'site', 'archaeological': 'site',
        'sculpture': 'artifact', 'painting': 'artifact', 'art': 'art'
    }
    
    for keyword, htype in heritage_keywords.items():
        if keyword in query_text:
            if htype not in concepts['heritage_types']:
                concepts['heritage_types'].append(htype)
            concepts['keywords'].append(keyword)
    
    # Domain keywords
    domain_keywords = {
        'religious': 'religious', 'temple': 'religious', 'mosque': 'religious',
        'church': 'religious', 'sacred': 'religious', 'spiritual': 'religious',
        'fort': 'military', 'fortress': 'military', 'defense': 'military',
        'palace': 'royal', 'king': 'royal', 'queen': 'royal', 'emperor': 'royal',
        'culture': 'cultural', 'tradition': 'cultural', 'festival': 'cultural',
        'archaeological': 'archaeological', 'excavation': 'archaeological',
        'architecture': 'architectural', 'building': 'architectural'
    }
    
    for keyword, domain in domain_keywords.items():
        if keyword in query_text:
            if domain not in concepts['domains']:
                concepts['domains'].append(domain)
    
    # Time period keywords
    period_keywords = {
        'ancient': 'ancient', 'prehistoric': 'ancient', 'old': 'ancient',
        'medieval': 'medieval', 'mughal': 'medieval', 'sultanate': 'medieval',
        'modern': 'modern', 'colonial': 'modern', 'british': 'modern', 'contemporary': 'modern'
    }
    
    for keyword, period in period_keywords.items():
        if keyword in query_text:
            if period not in concepts['time_periods']:
                concepts['time_periods'].append(period)
    
    # Architectural style keywords
    style_keywords = {
        'mughal': 'indo-islamic', 'sultanate': 'indo-islamic', 'dome': 'indo-islamic',
        'dravidian': 'dravidian', 'gopuram': 'dravidian',
        'buddhist': 'buddhist', 'stupa': 'buddhist',
        'colonial': 'colonial', 'gothic': 'colonial', 'victorian': 'colonial'
    }
    
    for keyword, style in style_keywords.items():
        if keyword in query_text:
            if style not in concepts['architectural_styles']:
                concepts['architectural_styles'].append(style)
    
    # Region keywords
    region_keywords = {
        'north india': 'north', 'delhi': 'north', 'rajasthan': 'north', 'agra': 'north',
        'south india': 'south', 'tamil nadu': 'south', 'karnataka': 'south',
        'east india': 'east', 'bengal': 'east', 'odisha': 'east',
        'west india': 'west', 'gujarat': 'west', 'maharashtra': 'west',
        'central india': 'central'
    }
    
    for keyword, region in region_keywords.items():
        if keyword in query_text:
            if region not in concepts['regions']:
                concepts['regions'].append(region)
    
    return concepts

def generate_query_ontology(query_text, heritage_ontology):
    """Generate ontology from user query"""
    
    concepts = parse_query(query_text)
    
    ontology = {
        'query': query_text,
        'concepts': {},
        'intent': 'search',  # Could be: search, explore, learn
        'constraints': {}
    }
    
    # Build concept hierarchies
    for htype in concepts['heritage_types']:
        hierarchy = heritage_ontology.get_concept_hierarchy(htype)
        if hierarchy:
            ontology['concepts']['HeritageType'] = ontology['concepts'].get('HeritageType', [])
            ontology['concepts']['HeritageType'].append({
                'value': htype,
                'hierarchy': hierarchy
            })
    
    for domain in concepts['domains']:
        hierarchy = heritage_ontology.get_concept_hierarchy(domain)
        if hierarchy:
            ontology['concepts']['Domain'] = ontology['concepts'].get('Domain', [])
            ontology['concepts']['Domain'].append({
                'value': domain,
                'hierarchy': hierarchy
            })
    
    for period in concepts['time_periods']:
        hierarchy = heritage_ontology.get_concept_hierarchy(period)
        if hierarchy:
            ontology['concepts']['TimePeriod'] = ontology['concepts'].get('TimePeriod', [])
            ontology['concepts']['TimePeriod'].append({
                'value': period,
                'hierarchy': hierarchy
            })
    
    if concepts['architectural_styles']:
        ontology['concepts']['ArchitecturalStyle'] = concepts['architectural_styles']
    
    if concepts['regions']:
        ontology['concepts']['GeographicRegion'] = concepts['regions']
    
    ontology['constraints']['keywords'] = concepts['keywords']
    
    return ontology

# ========== ONTOLOGY MATCHING ==========

def compute_ontology_similarity(query_ontology, doc_ontology):
    """Compute similarity between query and document ontologies"""
    
    score = 0.0
    max_score = 0.0
    
    # Match Heritage Types
    if 'HeritageType' in query_ontology['concepts'] and 'HeritageType' in doc_ontology['concepts']:
        query_types = [c['value'] for c in query_ontology['concepts']['HeritageType']]
        doc_types = doc_ontology['concepts']['HeritageType']
        
        matches = len(set(query_types) & set(doc_types))
        score += matches * 2.0  # High weight for heritage type match
        max_score += len(query_types) * 2.0
    
    # Match Domains
    if 'Domain' in query_ontology['concepts'] and 'Domain' in doc_ontology['concepts']:
        query_domains = [c['value'] for c in query_ontology['concepts']['Domain']]
        doc_domains = doc_ontology['concepts']['Domain']
        
        matches = len(set(query_domains) & set(doc_domains))
        score += matches * 1.5
        max_score += len(query_domains) * 1.5
    
    # Match Time Periods
    if 'TimePeriod' in query_ontology['concepts'] and 'TimePeriod' in doc_ontology['concepts']:
        query_periods = [c['value'] for c in query_ontology['concepts']['TimePeriod']]
        doc_periods = doc_ontology['concepts']['TimePeriod']
        
        matches = len(set(query_periods) & set(doc_periods))
        score += matches * 1.0
        max_score += len(query_periods) * 1.0
    
    # Match Architectural Styles
    if 'ArchitecturalStyle' in query_ontology['concepts'] and 'ArchitecturalStyle' in doc_ontology['concepts']:
        query_styles = query_ontology['concepts']['ArchitecturalStyle']
        doc_styles = doc_ontology['concepts']['ArchitecturalStyle']
        
        matches = len(set(query_styles) & set(doc_styles))
        score += matches * 1.0
        max_score += len(query_styles) * 1.0
    
    # Match Regions
    if 'GeographicRegion' in query_ontology['concepts'] and 'GeographicRegion' in doc_ontology['concepts']:
        query_regions = query_ontology['concepts']['GeographicRegion']
        doc_regions = doc_ontology['concepts']['GeographicRegion']
        
        matches = len(set(query_regions) & set(doc_regions))
        score += matches * 0.8
        max_score += len(query_regions) * 0.8
    
    # Normalize score
    if max_score > 0:
        return score / max_score
    else:
        return 0.0

# ========== MAIN EXECUTION ==========

def main():
    print("="*70)
    print("ONTOLOGY GENERATION SYSTEM")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize heritage ontology
    print("\n[Phase 1] Initializing Heritage Domain Ontology...")
    heritage_ontology = HeritageOntology()
    print(f"✓ Loaded ontology with {len(heritage_ontology.ontology)} top-level concepts")
    
    # Load documents
    print("\n[Phase 2] Loading classified documents...")
    with open(CLASSIFIED_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"✓ Loaded {len(documents)} documents")
    
    # Generate document ontologies
    print("\n[Phase 3] Generating document ontologies...")
    document_ontologies = []
    
    for idx, doc in enumerate(documents):
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(documents)} documents...")
        
        doc_ontology = generate_document_ontology(doc, heritage_ontology)
        document_ontologies.append(doc_ontology)
    
    print(f"✓ Generated {len(document_ontologies)} document ontologies")
    
    # Test query ontology generation
    print("\n[Phase 4] Testing query ontology generation...")
    test_queries = [
        "ancient temples in north India",
        "Mughal architecture forts",
        "Buddhist stupas and monasteries",
        "colonial buildings in Mumbai"
    ]
    
    query_ontologies = []
    for query in test_queries:
        query_ont = generate_query_ontology(query, heritage_ontology)
        query_ontologies.append(query_ont)
        print(f"\n  Query: '{query}'")
        print(f"    Concepts extracted: {list(query_ont['concepts'].keys())}")
    
    # Save ontologies
    print("\n[Phase 5] Saving ontologies...")
    os.makedirs(ONTOLOGY_DIR, exist_ok=True)
    
    ontology_data = {
        'heritage_ontology': heritage_ontology.ontology,
        'document_ontologies': document_ontologies,
        'test_query_ontologies': query_ontologies,
        'statistics': {
            'total_documents': len(document_ontologies),
            'ontology_concepts': len(heritage_ontology.ontology),
            'avg_concepts_per_doc': sum(len(d['concepts']) for d in document_ontologies) / len(document_ontologies)
        },
        'generation_date': datetime.now().isoformat()
    }
    
    with open(ONTOLOGY_FILE, 'w', encoding='utf-8') as f:
        json.dump(ontology_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Ontologies saved to: {ONTOLOGY_FILE}")
    
    # Summary
    print("\n" + "="*70)
    print("ONTOLOGY GENERATION COMPLETE")
    print("="*70)
    print(f"✅ Generated ontologies for {len(document_ontologies)} documents")
    print(f"✅ Domain ontology covers {len(heritage_ontology.ontology)} concept hierarchies")
    print(f"📊 Average concepts per document: {ontology_data['statistics']['avg_concepts_per_doc']:.2f}")
    print(f"💾 Saved to: {ONTOLOGY_FILE}")
    print("="*70)

if __name__ == "__main__":
    main()