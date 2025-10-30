import wikipedia
import json
import os
from datetime import datetime

# Use Wikipedia to find historical heritage documents and books
HERITAGE_DOCUMENT_TOPICS = [
    "History of Indian architecture",
    "Archaeological Survey of India",
    "Conservation of cultural heritage",
    "Ancient Indian architecture",
    "Medieval Indian architecture",
    "Mughal architecture",
    "Buddhist architecture",
    "Hindu temple architecture",
    "Indo-Islamic architecture",
    "Rock-cut architecture in India",
    "Heritage conservation in India",
    "Ancient monuments of India",
    "Archaeological sites in India",
    "Indian architectural history",
    "Historical monuments of India"
]

def fetch_heritage_document(topic):
    """Fetch Wikipedia article about heritage documentation/history"""
    try:
        page = wikipedia.page(topic, auto_suggest=True)
        
        article_data = {
            'title': page.title,
            'url': page.url,
            'content': page.content,
            'summary': page.summary,
            'categories': page.categories,
            'source': 'Heritage Documentation (Wikipedia)',
            'metadata': {
                'document_type': 'Historical/Documentary',
                'topic': topic
            },
            'fetched_at': datetime.now().isoformat()
        }
        
        return article_data
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

def main():
    print("="*60)
    print("HERITAGE DOCUMENTATION SCRAPER")
    print("="*60)
    print("\nCollecting historical and documentary sources...\n")
    
    articles = []
    
    for idx, topic in enumerate(HERITAGE_DOCUMENT_TOPICS, 1):
        print(f"[{idx}/{len(HERITAGE_DOCUMENT_TOPICS)}] {topic}")
        
        article = fetch_heritage_document(topic)
        if article:
            articles.append(article)
            print(f"  ✓ {article['title']}")
    
    # Save articles
    print(f"\n[Saving] {len(articles)} heritage documents...")
    os.makedirs('data/raw/archives', exist_ok=True)
    
    for i, article in enumerate(articles):
        safe_title = "".join(c for c in article['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"doc_{i+1:03d}_{safe_title[:50]}.json"
        filepath = os.path.join('data/raw/archives', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("HERITAGE DOCUMENTATION COMPLETE")
    print("="*60)
    print(f"Documents collected: {len(articles)}")
    print(f"Saved to: data/raw/archives/")

if __name__ == "__main__":
    main()