import wikipedia
import json
import os
import time
from datetime import datetime
from bs4 import BeautifulSoup

def get_unesco_sites_from_wikipedia():
    """Get UNESCO sites from Wikipedia's comprehensive list"""
    print("\n" + "="*60)
    print("UNESCO WORLD HERITAGE SCRAPER (via Wikipedia)")
    print("="*60)
    
    try:
        # Get the master list page
        print("\n[1/2] Fetching UNESCO World Heritage Sites list from Wikipedia...")
        page = wikipedia.page("List of World Heritage Sites", auto_suggest=False)
        
        soup = BeautifulSoup(page.html(), 'html.parser')
        
        # Find all links in tables (UNESCO sites are in tables)
        sites = []
        tables = soup.find_all('table', class_='wikitable')
        
        for table in tables:
            links = table.find_all('a', href=True)
            for link in links:
                title = link.get('title', '')
                # Filter for actual heritage sites
                if title and not any(skip in title.lower() for skip in 
                    ['list of', 'world heritage site', 'unesco', 'edit', 'citation', 
                     'help:', 'wikipedia:', 'category:', 'file:', 'template:']):
                    sites.append(title)
        
        # Remove duplicates
        sites = list(set(sites))
        print(f"✓ Found {len(sites)} potential UNESCO sites")
        
        return sites[:60]  # Limit to 60 for time
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return []

def fetch_unesco_site_details(site_name):
    """Fetch details for a UNESCO heritage site"""
    try:
        page = wikipedia.page(site_name, auto_suggest=False)
        
        # Check if it's actually a UNESCO site
        content = page.content.lower()
        if 'unesco' not in content and 'world heritage' not in content:
            return None
        
        article_data = {
            'title': page.title,
            'url': page.url,
            'content': page.content,
            'summary': page.summary,
            'categories': page.categories,
            'source': 'UNESCO World Heritage (Wikipedia)',
            'metadata': {
                'heritage_type': 'UNESCO World Heritage Site',
                'verified_unesco': 'unesco' in content or 'world heritage' in content
            },
            'fetched_at': datetime.now().isoformat()
        }
        
        return article_data
        
    except wikipedia.exceptions.DisambiguationError:
        print(f"  ⚠ Disambiguation: {site_name}")
        return None
    except wikipedia.exceptions.PageError:
        print(f"  ⚠ Page not found: {site_name}")
        return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

def main():
    # Get UNESCO sites list
    sites = get_unesco_sites_from_wikipedia()
    
    if not sites:
        print("✗ No UNESCO sites found. Exiting.")
        return
    
    print(f"\n[2/2] Fetching details for {len(sites)} UNESCO sites...")
    print("This will take ~10-15 minutes...\n")
    
    articles = []
    
    for idx, site_name in enumerate(sites, 1):
        print(f"[{idx}/{len(sites)}] {site_name}")
        
        article = fetch_unesco_site_details(site_name)
        if article:
            articles.append(article)
            print(f"  ✓ {article['title']}")
        
        time.sleep(1.5)  # Rate limiting
    
    # Save articles
    print(f"\n[Saving] {len(articles)} UNESCO heritage documents...")
    os.makedirs('data/raw/unesco', exist_ok=True)
    
    for i, article in enumerate(articles):
        safe_title = "".join(c for c in article['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"unesco_{i+1:03d}_{safe_title[:50]}.json"
        filepath = os.path.join('data/raw/unesco', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("UNESCO COLLECTION COMPLETE")
    print("="*60)
    print(f"Sites scanned: {len(sites)}")
    print(f"Documents collected: {len(articles)}")
    print(f"Success rate: {len(articles)/len(sites)*100:.1f}%")
    print(f"Saved to: data/raw/unesco/")

if __name__ == "__main__":
    main()