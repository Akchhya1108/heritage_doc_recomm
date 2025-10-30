"""
STEP 2: ENHANCED METADATA EXTRACTION
Extracts rich metadata for Knowledge Graph construction
"""

import json
import os
import re
from datetime import datetime
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('stopwords')

# Directories
CLEAN_DIR = "data/cleaned data"
META_DIR = "data/metadata"
OUTPUT_FILE = os.path.join(META_DIR, "enriched_metadata.json")

# ========== CLASSIFICATION RULES ==========

HERITAGE_TYPES = {
    'monument': ['temple', 'fort', 'palace', 'monument', 'memorial', 'tomb', 'mosque', 'church', 'stupa', 'tower', 'gate', 'wall'],
    'site': ['site', 'complex', 'ruins', 'excavation', 'settlement', 'city', 'town', 'village'],
    'artifact': ['sculpture', 'statue', 'painting', 'manuscript', 'inscription', 'coin', 'pottery', 'artifact'],
    'architecture': ['architecture', 'building', 'structure', 'construction', 'design', 'style'],
    'tradition': ['tradition', 'festival', 'ritual', 'custom', 'practice', 'dance', 'music', 'craft'],
    'art': ['art', 'carving', 'mural', 'fresco', 'relief', 'iconography']
}

DOMAINS = {
    'religious': ['temple', 'mosque', 'church', 'monastery', 'shrine', 'worship', 'buddhist', 'hindu', 'islam', 'christian', 'jain', 'sikh', 'religious', 'sacred', 'spiritual', 'deity', 'god', 'prayer'],
    'military': ['fort', 'fortress', 'defense', 'battle', 'war', 'army', 'military', 'garrison', 'citadel', 'rampart'],
    'royal': ['palace', 'king', 'queen', 'emperor', 'sultan', 'maharaja', 'royal', 'court', 'throne', 'dynasty'],
    'cultural': ['culture', 'festival', 'tradition', 'heritage', 'art', 'music', 'dance', 'literature'],
    'archaeological': ['archaeological', 'excavation', 'ruins', 'ancient', 'prehistoric', 'neolithic', 'bronze age'],
    'architectural': ['architecture', 'design', 'construction', 'building', 'engineering', 'structural']
}

TIME_PERIODS = {
    'ancient': ['ancient', 'prehistoric', 'indus valley', 'vedic', 'maurya', 'gupta', 'classical', 'bce', 'bc'],
    'medieval': ['medieval', 'sultanate', 'mughal', 'vijayanagar', 'chola', 'pallava', 'rashtrakuta', 'rajput', '10th century', '11th century', '12th century', '13th century', '14th century', '15th century', '16th century'],
    'modern': ['modern', 'colonial', 'british', 'contemporary', 'independence', '17th century', '18th century', '19th century', '20th century', '21st century']
}

ARCHITECTURAL_STYLES = {
    'indo-islamic': ['indo-islamic', 'mughal', 'sultanate', 'dome', 'minaret', 'arch', 'persian'],
    'dravidian': ['dravidian', 'gopuram', 'vimana', 'mandapa', 'south indian'],
    'nagara': ['nagara', 'shikhara', 'north indian', 'rekha-deul'],
    'vesara': ['vesara', 'hoysala', 'chalukya'],
    'buddhist': ['buddhist', 'stupa', 'chaitya', 'vihara', 'monastery'],
    'colonial': ['colonial', 'british', 'gothic', 'victorian', 'european']
}

INDIAN_REGIONS = {
    'north': ['delhi', 'punjab', 'haryana', 'uttar pradesh', 'rajasthan', 'jammu', 'kashmir', 'himachal', 'uttarakhand'],
    'south': ['tamil nadu', 'karnataka', 'kerala', 'andhra pradesh', 'telangana'],
    'east': ['west bengal', 'odisha', 'bihar', 'jharkhand', 'assam', 'meghalaya'],
    'west': ['gujarat', 'maharashtra', 'goa'],
    'central': ['madhya pradesh', 'chhattisgarh']
}

# ========== EXTRACTION FUNCTIONS ==========

def extract_named_entities(text):
    """Extract named entities using NLTK"""
    entities = {
        'persons': [],
        'locations': [],
        'organizations': [],
        'dates': []
    }
    
    try:
        sentences = sent_tokenize(text[:5000])  # First 5000 chars for speed
        
        for sentence in sentences[:20]:  # First 20 sentences
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            chunks = ne_chunk(tagged, binary=False)
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity = ' '.join(c[0] for c in chunk)
                    
                    if chunk.label() == 'PERSON':
                        entities['persons'].append(entity)
                    elif chunk.label() == 'GPE' or chunk.label() == 'LOCATION':
                        entities['locations'].append(entity)
                    elif chunk.label() == 'ORGANIZATION':
                        entities['organizations'].append(entity)
        
        # Extract dates with regex
        date_pattern = r'\b(\d{1,4}\s*(AD|BCE?|CE)\b|\d{4}s?\b|\d{1,2}th\s+century)'
        entities['dates'] = re.findall(date_pattern, text, re.IGNORECASE)
        entities['dates'] = [d[0] if isinstance(d, tuple) else d for d in entities['dates']]
        
    except Exception as e:
        print(f"  ⚠ NER Error: {e}")
    
    # Remove duplicates and clean
    for key in entities:
        entities[key] = list(set([e.strip() for e in entities[key] if len(e.strip()) > 2]))
    
    return entities

def classify_heritage_type(text):
    """Classify document into heritage type"""
    text_lower = text.lower()
    scores = {htype: 0 for htype in HERITAGE_TYPES}
    
    for htype, keywords in HERITAGE_TYPES.items():
        for keyword in keywords:
            scores[htype] += text_lower.count(keyword)
    
    # Return top 2 types
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_types[:2] if t[1] > 0]

def classify_domain(text):
    """Classify into cultural domain"""
    text_lower = text.lower()
    scores = {domain: 0 for domain in DOMAINS}
    
    for domain, keywords in DOMAINS.items():
        for keyword in keywords:
            scores[domain] += text_lower.count(keyword)
    
    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [d[0] for d in sorted_domains[:3] if d[1] > 0]

def classify_time_period(text):
    """Classify into time period"""
    text_lower = text.lower()
    scores = {period: 0 for period in TIME_PERIODS}
    
    for period, keywords in TIME_PERIODS.items():
        for keyword in keywords:
            scores[period] += text_lower.count(keyword)
    
    if scores['ancient'] > 0:
        return 'ancient'
    elif scores['medieval'] > 0:
        return 'medieval'
    elif scores['modern'] > 0:
        return 'modern'
    return 'unknown'

def extract_architectural_style(text):
    """Extract architectural style"""
    text_lower = text.lower()
    found_styles = []
    
    for style, keywords in ARCHITECTURAL_STYLES.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_styles.append(style)
                break
    
    return list(set(found_styles))

def classify_region(text):
    """Classify Indian region"""
    text_lower = text.lower()
    
    for region, states in INDIAN_REGIONS.items():
        for state in states:
            if state in text_lower:
                return region
    
    # Check for country mentions
    if 'india' in text_lower:
        return 'india'
    
    return 'unknown'

def extract_keywords_tfidf(documents, top_n=10):
    """Extract keywords using TF-IDF"""
    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top keywords per document
        all_keywords = []
        for doc_idx in range(len(documents)):
            scores = tfidf_matrix[doc_idx].toarray()[0]
            top_indices = scores.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
            all_keywords.append(keywords)
        
        return all_keywords
    except Exception as e:
        print(f"⚠ TF-IDF Error: {e}")
        return [[] for _ in documents]

def determine_tangibility(heritage_types):
    """Determine if heritage is tangible or intangible"""
    tangible = ['monument', 'site', 'artifact', 'architecture']
    intangible = ['tradition', 'art']
    
    if any(t in tangible for t in heritage_types):
        return 'tangible'
    elif any(t in intangible for t in heritage_types):
        return 'intangible'
    return 'tangible'  # default

# ========== MAIN PROCESSING ==========

def process_all_documents():
    """Process all cleaned documents and extract metadata"""
    
    print("="*70)
    print("ENHANCED METADATA EXTRACTION")
    print("="*70)
    
    # Load existing metadata
    meta_file = os.path.join(META_DIR, "metadata.json")
    
    if not os.path.exists(meta_file):
        print("✗ metadata.json not found! Run clean_data.py first.")
        return
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"\n📄 Found {len(metadata)} documents")
    print("\n[Phase 1] Loading document texts...")
    
    # Load all document texts
    documents = []
    valid_metadata = []
    
    for meta in metadata:
        cleaned_path = meta.get('cleaned_path', '')
        if os.path.exists(cleaned_path):
            try:
                with open(cleaned_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    documents.append(text)
                    valid_metadata.append(meta)
            except Exception as e:
                print(f"⚠ Could not read {cleaned_path}: {e}")
    
    print(f"✓ Loaded {len(documents)} document texts")
    
    # Extract TF-IDF keywords for all documents
    print("\n[Phase 2] Extracting TF-IDF keywords...")
    all_keywords = extract_keywords_tfidf(documents, top_n=15)
    print("✓ TF-IDF extraction complete")
    
    # Process each document
    print("\n[Phase 3] Extracting rich metadata...")
    enriched_metadata = []
    
    for idx, (meta, text) in enumerate(zip(valid_metadata, documents), 1):
        print(f"[{idx}/{len(valid_metadata)}] {meta['title'][:50]}...")
        
        # Extract entities
        entities = extract_named_entities(text)
        
        # Classifications
        heritage_types = classify_heritage_type(text)
        domains = classify_domain(text)
        time_period = classify_time_period(text)
        arch_styles = extract_architectural_style(text)
        region = classify_region(text)
        tangibility = determine_tangibility(heritage_types)
        
        # Build enriched metadata
        enriched = {
            **meta,  # Keep original metadata
            'entities': entities,
            'classifications': {
                'heritage_types': heritage_types,
                'domains': domains,
                'time_period': time_period,
                'architectural_styles': arch_styles,
                'region': region,
                'tangibility': tangibility
            },
            'keywords_tfidf': all_keywords[idx-1],
            'enrichment_date': datetime.now().isoformat()
        }
        
        enriched_metadata.append(enriched)
    
    # Save enriched metadata
    print(f"\n[Phase 4] Saving enriched metadata...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(enriched_metadata, f, indent=2, ensure_ascii=False)
    
    # Statistics
    print("\n" + "="*70)
    print("METADATA EXTRACTION COMPLETE")
    print("="*70)
    print(f"✅ Processed: {len(enriched_metadata)} documents")
    print(f"📊 Output: {OUTPUT_FILE}")
    
    # Show sample statistics
    all_types = [t for doc in enriched_metadata for t in doc['classifications']['heritage_types']]
    all_domains = [d for doc in enriched_metadata for d in doc['classifications']['domains']]
    time_periods = [doc['classifications']['time_period'] for doc in enriched_metadata]
    
    print("\n📈 STATISTICS:")
    print(f"  Heritage Types: {dict(Counter(all_types).most_common(5))}")
    print(f"  Domains: {dict(Counter(all_domains).most_common(5))}")
    print(f"  Time Periods: {dict(Counter(time_periods))}")
    
    print("\n💡 Sample document structure:")
    if enriched_metadata:
        sample = enriched_metadata[0]
        print(f"  Title: {sample['title']}")
        print(f"  Heritage Types: {sample['classifications']['heritage_types']}")
        print(f"  Domains: {sample['classifications']['domains']}")
        print(f"  Time Period: {sample['classifications']['time_period']}")
        print(f"  Region: {sample['classifications']['region']}")
        print(f"  Entities Found:")
        print(f"    - Locations: {len(sample['entities']['locations'])}")
        print(f"    - Persons: {len(sample['entities']['persons'])}")
        print(f"    - Organizations: {len(sample['entities']['organizations'])}")
        print(f"  Keywords (top 5): {sample['keywords_tfidf'][:5]}")
    
    print("\n✅ Ready for Step 3: Generate Embeddings!")
    print("="*70)

if __name__ == "__main__":
    process_all_documents()