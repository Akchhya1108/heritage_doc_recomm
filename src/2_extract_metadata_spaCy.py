import json
import os
import re
from datetime import datetime
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Directories
CLEAN_DIR = "data/cleaned data"
META_DIR = "data/metadata"
OUTPUT_FILE = os.path.join(META_DIR, "enriched_metadata.json")

# ========== INDIAN HERITAGE ENTITY LISTS (Custom) ==========

# Common Indian heritage locations
INDIAN_LOCATIONS = {
    'taj mahal', 'red fort', 'qutub minar', 'india gate', 'gateway of india',
    'ajanta caves', 'ellora caves', 'khajuraho', 'hampi', 'fatehpur sikri',
    'konark', 'sanchi', 'nalanda', 'bodh gaya', 'varanasi', 'madurai',
    'thanjavur', 'mahabalipuram', 'pattadakal', 'aihole', 'badami',
    'mehrangarh', 'amber fort', 'mysore palace', 'victoria memorial',
    'charminar', 'golconda', 'agra', 'delhi', 'jaipur', 'udaipur',
    'mumbai', 'kolkata', 'chennai', 'bangalore', 'hyderabad',
    'rajasthan', 'maharashtra', 'karnataka', 'tamil nadu', 'kerala',
    'gujarat', 'madhya pradesh', 'uttar pradesh', 'bihar', 'odisha'
}

# Historical figures
INDIAN_PERSONS = {
    'ashoka', 'akbar', 'shah jahan', 'aurangzeb', 'babur', 'humayun',
    'krishnadevaraya', 'rajaraja chola', 'raja raja', 'rajendra chola',
    'chandragupta', 'samudragupta', 'harsha', 'pulakeshin',
    'shivaji', 'rani lakshmibai', 'tipu sultan', 'maharana pratap',
    'gautama buddha', 'mahavira', 'guru nanak', 'kabir',
    'kalidasa', 'aryabhata', 'charaka', 'sushruta'
}

# Dynasties and organizations
INDIAN_ORGANIZATIONS = {
    'mughal', 'maurya', 'gupta', 'chola', 'pallava', 'chalukya',
    'hoysala', 'vijayanagara', 'maratha', 'rajput', 'sultanate',
    'archaeological survey of india', 'asi', 'unesco',
    'indian national trust', 'intach'
}

# Monument types
MONUMENT_KEYWORDS = {
    'temple', 'fort', 'palace', 'mosque', 'church', 'stupa', 'monastery',
    'tomb', 'mausoleum', 'memorial', 'gateway', 'gate', 'tower', 'minaret',
    'cave', 'complex', 'site', 'ruins', 'monument'
}

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

# ========== ENTITY EXTRACTION WITH spaCy + Custom Lists ==========

def extract_entities_with_spacy(text, title):
    """
    Extract named entities using spaCy + custom Indian heritage lists
    This is MUCH faster and more reliable than Gemini!
    """
    
    entities = {
        'locations': [],
        'persons': [],
        'organizations': [],
        'dates': [],
        'monuments': []
    }
    
    # Process with spaCy (first 5000 chars for speed)
    doc = nlp(text[:5000])
    
    # Extract entities using spaCy NER
    for ent in doc.ents:
        entity_text = ent.text.strip()
        entity_lower = entity_text.lower()
        
        if ent.label_ == "GPE" or ent.label_ == "LOC":
            # Geographic/Location entity
            entities['locations'].append(entity_text)
            
        elif ent.label_ == "PERSON":
            # Person entity
            entities['persons'].append(entity_text)
            
        elif ent.label_ == "ORG":
            # Organization entity
            entities['organizations'].append(entity_text)
            
        elif ent.label_ == "DATE":
            # Date entity
            entities['dates'].append(entity_text)
    
    # Enhance with custom Indian heritage lists
    text_lower = text.lower()
    
    # Extract known Indian locations
    for location in INDIAN_LOCATIONS:
        if location in text_lower and location not in [l.lower() for l in entities['locations']]:
            entities['locations'].append(location.title())
    
    # Extract known Indian persons
    for person in INDIAN_PERSONS:
        if person in text_lower and person not in [p.lower() for p in entities['persons']]:
            entities['persons'].append(person.title())
    
    # Extract known organizations/dynasties
    for org in INDIAN_ORGANIZATIONS:
        if org in text_lower and org not in [o.lower() for o in entities['organizations']]:
            entities['organizations'].append(org.title())
    
    # Extract monuments (buildings with monument keywords)
    # Look for patterns like "X temple", "Y fort", "Z palace"
    for keyword in MONUMENT_KEYWORDS:
        # Find mentions like "Brihadeeswarar Temple", "Red Fort", etc.
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+' + keyword
        matches = re.findall(pattern, text[:3000], re.IGNORECASE)
        for match in matches:
            monument_name = f"{match} {keyword}"
            if monument_name not in entities['monuments']:
                entities['monuments'].append(monument_name.title())
    
    # Extract dates with regex (centuries, years, etc.)
    date_patterns = [
        r'\b(\d{1,4}\s*(?:AD|BCE?|CE))\b',
        r'\b(\d{4}s?)\b',
        r'\b(\d{1,2}th\s+century)\b',
        r'\b(century\s+(?:AD|BCE?|CE))\b'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            if match and match not in entities['dates']:
                entities['dates'].append(match)
    
    # Remove duplicates and clean
    for key in entities:
        # Remove duplicates (case-insensitive)
        seen = set()
        unique = []
        for item in entities[key]:
            item_lower = item.lower().strip()
            if item_lower and len(item_lower) > 2 and item_lower not in seen:
                seen.add(item_lower)
                unique.append(item.strip())
        entities[key] = unique[:20]  # Limit to top 20 per category
    
    return entities

# ========== CLASSIFICATION FUNCTIONS ==========

def classify_heritage_type(text):
    text_lower = text.lower()
    scores = {htype: 0 for htype in HERITAGE_TYPES}
    
    for htype, keywords in HERITAGE_TYPES.items():
        for keyword in keywords:
            scores[htype] += text_lower.count(keyword)
    
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_types[:2] if t[1] > 0]

def classify_domain(text):
    text_lower = text.lower()
    scores = {domain: 0 for domain in DOMAINS}
    
    for domain, keywords in DOMAINS.items():
        for keyword in keywords:
            scores[domain] += text_lower.count(keyword)
    
    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [d[0] for d in sorted_domains[:3] if d[1] > 0]

def classify_time_period(text):
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
    text_lower = text.lower()
    found_styles = []
    
    for style, keywords in ARCHITECTURAL_STYLES.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_styles.append(style)
                break
    
    return list(set(found_styles))

def classify_region(text):
    text_lower = text.lower()
    
    for region, states in INDIAN_REGIONS.items():
        for state in states:
            if state in text_lower:
                return region
    
    if 'india' in text_lower:
        return 'india'
    
    return 'unknown'

def extract_keywords_tfidf(documents, top_n=10):
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
    tangible = ['monument', 'site', 'artifact', 'architecture']
    intangible = ['tradition', 'art']
    
    if any(t in tangible for t in heritage_types):
        return 'tangible'
    elif any(t in intangible for t in heritage_types):
        return 'intangible'
    return 'tangible'

# ========== MAIN PROCESSING ==========

def process_all_documents():
    print("="*70)
    print("ENHANCED METADATA EXTRACTION WITH spaCy")
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
    
    # Extract TF-IDF keywords
    print("\n[Phase 2] Extracting TF-IDF keywords...")
    all_keywords = extract_keywords_tfidf(documents, top_n=15)
    print("✓ TF-IDF extraction complete")
    
    # Process each document with spaCy
    print("\n[Phase 3] Extracting entities with spaCy...")
    print("⏱️  This will take ~2-3 minutes for 369 documents")
    
    enriched_metadata = []
    
    for idx, (meta, text) in enumerate(zip(valid_metadata, documents), 1):
        if idx % 50 == 0 or idx == 1:
            print(f"\n[{idx}/{len(valid_metadata)}] {meta['title'][:60]}...")
        
        # Extract entities using spaCy + custom lists
        entities = extract_entities_with_spacy(text, meta['title'])
        
        # Show progress
        if idx % 50 == 0 or idx == 1:
            total_entities = sum(len(entities[key]) for key in entities)
            print(f"    ✓ Extracted {total_entities} entities: "
                  f"{len(entities['locations'])} locations, "
                  f"{len(entities['persons'])} persons, "
                  f"{len(entities['organizations'])} orgs, "
                  f"{len(entities['monuments'])} monuments")
        
        # Classifications
        heritage_types = classify_heritage_type(text)
        domains = classify_domain(text)
        time_period = classify_time_period(text)
        arch_styles = extract_architectural_style(text)
        region = classify_region(text)
        tangibility = determine_tangibility(heritage_types)
        
        # Build enriched metadata
        enriched = {
            **meta,
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
    print("✅ METADATA EXTRACTION COMPLETE")
    print("="*70)
    print(f"Processed: {len(enriched_metadata)} documents")
    print(f"Output: {OUTPUT_FILE}")
    
    # Entity statistics
    total_locations = sum(len(doc['entities']['locations']) for doc in enriched_metadata)
    total_persons = sum(len(doc['entities']['persons']) for doc in enriched_metadata)
    total_orgs = sum(len(doc['entities']['organizations']) for doc in enriched_metadata)
    total_monuments = sum(len(doc['entities']['monuments']) for doc in enriched_metadata)
    
    print("\n📊 ENTITY STATISTICS:")
    print(f"  Total Locations: {total_locations}")
    print(f"  Total Persons: {total_persons}")
    print(f"  Total Organizations: {total_orgs}")
    print(f"  Total Monuments: {total_monuments}")
    print(f"  Average entities per doc: {(total_locations + total_persons + total_orgs + total_monuments) / len(enriched_metadata):.1f}")
    
    # Classification statistics
    all_types = [t for doc in enriched_metadata for t in doc['classifications']['heritage_types']]
    all_domains = [d for doc in enriched_metadata for d in doc['classifications']['domains']]
    time_periods = [doc['classifications']['time_period'] for doc in enriched_metadata]
    
    print("\n📈 CLASSIFICATION STATISTICS:")
    print(f"  Heritage Types: {dict(Counter(all_types).most_common(5))}")
    print(f"  Domains: {dict(Counter(all_domains).most_common(5))}")
    print(f"  Time Periods: {dict(Counter(time_periods))}")
    
    # Sample
    print("\n💡 SAMPLE DOCUMENT:")
    if enriched_metadata:
        sample = enriched_metadata[0]
        print(f"  Title: {sample['title']}")
        print(f"  Heritage Types: {sample['classifications']['heritage_types']}")
        print(f"  Entities:")
        print(f"    Locations: {sample['entities']['locations'][:5]}")
        print(f"    Persons: {sample['entities']['persons'][:3]}")
        print(f"    Monuments: {sample['entities']['monuments'][:3]}")
    
    print("="*70)

if __name__ == "__main__":
    process_all_documents()