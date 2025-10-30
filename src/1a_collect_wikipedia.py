import wikipedia
import json
import os
import time
from datetime import datetime

# Heritage-related search queries
HERITAGE_QUERIES = [
    "UNESCO World Heritage Site",
    "Ancient monuments India",
    "Historical architecture",
    "Cultural heritage",
    "Archaeological sites",
    "Historical monuments",
    "Heritage buildings",
    "Ancient temples",
    "Historical forts",
    "Palace architecture",
    "Ancient civilizations",
    "Historical landmarks",
    "Heritage conservation",
    "Traditional art forms",
    "Ancient sculptures",
    "Historical artifacts",
    "Medieval architecture",
    "Colonial architecture",
    "Religious heritage sites",
    "Ancient ruins"
]

def fetch_wikipedia_articles(query, max_articles=8):
    """Fetch Wikipedia articles for a given query"""
    articles = []
    try:
        # Search for related pages
        search_results = wikipedia.search(query, results=max_articles)
        
        for title in search_results:
            try:
                # Fetch page content
                page = wikipedia.page(title, auto_suggest=False)
                
                article_data = {
                    'title': page.title,
                    'url': page.url,
                    'content': page.content,
                    'summary': page.summary,
                    'categories': page.categories,
                    'query': query,
                    'fetched_at': datetime.now().isoformat()
                }
                
                articles.append(article_data)
                print(f"✓ Fetched: {page.title}")
                time.sleep(0.5)  # Be nice to Wikipedia
                
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"⚠ Disambiguation for {title}, skipping...")
                continue
            except Exception as e:
                print(f"✗ Error fetching {title}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"✗ Search error for '{query}': {str(e)}")
    
    return articles

def save_articles(articles, output_dir='data/raw'):
    """Save articles to JSON files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, article in enumerate(articles):
        # Create safe filename
        safe_title = "".join(c for c in article['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{i+1:03d}_{safe_title[:50]}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(articles)} articles to {output_dir}")

def main():
    print("=" * 60)
    print("HERITAGE DOCUMENT COLLECTOR")
    print("=" * 60)
    print(f"\nTarget: 100-150 documents")
    print(f"Queries: {len(HERITAGE_QUERIES)}")
    print("\nStarting collection...\n")
    
    all_articles = []
    seen_titles = set()  # Avoid duplicates
    
    for idx, query in enumerate(HERITAGE_QUERIES, 1):
        print(f"\n[{idx}/{len(HERITAGE_QUERIES)}] Searching: '{query}'")
        print("-" * 60)
        
        articles = fetch_wikipedia_articles(query, max_articles=8)
        
        # Filter duplicates
        for article in articles:
            if article['title'] not in seen_titles:
                all_articles.append(article)
                seen_titles.add(article['title'])
        
        print(f"Total unique articles: {len(all_articles)}")
        
        # Stop if we have enough
        if len(all_articles) >= 150:
            print("\n✓ Reached target of 150 documents!")
            break
        
        time.sleep(1)  # Rate limiting
    
    # Save everything
    save_articles(all_articles)
    
    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total documents collected: {len(all_articles)}")
    print(f"Saved to: data/raw/")
    print("\nSample titles:")
    for article in all_articles[:5]:
        print(f"  - {article['title']}")

if __name__ == "__main__":
    main()