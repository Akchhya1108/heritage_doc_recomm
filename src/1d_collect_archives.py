import requests
import json
import os
from datetime import datetime

def search_archive_org(query, num_results=5):
    """Search Archive.org for heritage documents"""
    base_url = "https://archive.org/advancedsearch.php"
    
    params = {
        'q': query,
        'fl[]': ['identifier', 'title', 'description', 'subject', 'creator', 'date'],
        'rows': num_results,
        'page': 1,
        'output': 'json',
        'mediatype': 'texts'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        return data.get('response', {}).get('docs', [])
    except Exception as e:
        print(f"✗ Error searching Archive.org for '{query}': {e}")
        return []

def fetch_archive_metadata(identifier):
    """Fetch detailed metadata for an Archive.org item"""
    url = f"https://archive.org/metadata/{identifier}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"✗ Error fetching metadata for {identifier}: {e}")
        return None

ARCHIVE_QUERIES = [
    "heritage architecture",
    "ancient monuments",
    "historical buildings",
    "archaeological sites",
    "cultural heritage",
    "world heritage",
    "historical preservation",
    "traditional architecture"
]

def main():
    print("="*60)
    print("ARCHIVE.ORG HERITAGE DOCUMENTS SCRAPER")
    print("="*60)
    print("\nSearching historical documents...\n")
    
    all_articles = []
    seen_ids = set()
    
    for idx, query in enumerate(ARCHIVE_QUERIES, 1):
        print(f"[{idx}/{len(ARCHIVE_QUERIES)}] Searching: '{query}'")
        
        results = search_archive_org(query, num_results=3)
        
        for result in results:
            identifier = result.get('identifier')
            
            if identifier in seen_ids:
                continue
            seen_ids.add(identifier)
            
            title = result.get('title', 'Unknown')
            description = result.get('description', '')
            
            article_data = {
                'title': title if isinstance(title, str) else title[0] if title else 'Unknown',
                'url': f"https://archive.org/details/{identifier}",
                'content': description if isinstance(description, str) else ' '.join(description) if description else '',
                'summary': (description[:300] + "...") if isinstance(description, str) and len(description) > 300 else description,
                'source': 'Archive.org',
                'metadata': {
                    'identifier': identifier,
                    'subject': result.get('subject', []),
                    'creator': result.get('creator', ''),
                    'date': result.get('date', '')
                },
                'query': query,
                'fetched_at': datetime.now().isoformat()
            }
            
            all_articles.append(article_data)
            print(f"✓ {article_data['title'][:60]}")
    
    # Save articles
    print(f"\nSaving {len(all_articles)} articles...")
    os.makedirs('data/raw/archives', exist_ok=True)
    
    for i, article in enumerate(all_articles):
        safe_title = "".join(c for c in str(article['title']) if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"archive_{i+1:03d}_{safe_title[:50]}.json"
        filepath = os.path.join('data/raw/archives', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("ARCHIVE.ORG COLLECTION COMPLETE")
    print("="*60)
    print(f"Total documents: {len(all_articles)}")
    print(f"Saved to: data/raw/archives/")

if __name__ == "__main__":
    main()