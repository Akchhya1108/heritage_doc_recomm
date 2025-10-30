import requests
from bs4 import BeautifulSoup
import json
import os
import time
from datetime import datetime
import wikipedia

def scrape_indian_heritage_lists():
    """Scrape Wikipedia lists of Indian heritage sites"""
    print("\n" + "="*60)
    print("DISCOVERING INDIAN HERITAGE SITES")
    print("="*60)
    
    # Wikipedia list pages
    list_pages = [
        "List of World Heritage Sites in India",
        "List of Monuments of National Importance",
        "Archaeological sites in India"
    ]
    
    all_sites = []
    
    for list_page in list_pages:
        print(f"\n[Scanning] {list_page}")
        try:
            page = wikipedia.page(list_page, auto_suggest=False)
            soup = BeautifulSoup(page.html(), 'html.parser')
            
            # Find all links in tables (where heritage sites are usually listed)
            tables = soup.find_all('table', class_='wikitable')
            
            for table in tables:
                links = table.find_all('a', href=True)
                for link in links:
                    title = link.get('title', '')
                    # Filter out non-relevant links
                    if title and not any(skip in title.lower() for skip in ['list of', 'edit', 'citation', 'help:', 'wikipedia:']):
                        all_sites.append(title)
            
            print(f"✓ Found {len(all_sites)} potential sites so far")
            time.sleep(1)
            
        except Exception as e:
            print(f"⚠ Could not parse {list_page}: {e}")
            continue
    
    # Remove duplicates and clean
    all_sites = list(set(all_sites))
    print(f"\n✓ Total unique sites discovered: {len(all_sites)}")
    
    return all_sites

def fetch_heritage_article(site_name):
    """Fetch Wikipedia article for heritage site"""
    try:
        page = wikipedia.page(site_name, auto_suggest=False)
        
        # Check if it's actually a heritage site (has relevant keywords)
        content_lower = page.content.lower()
        heritage_keywords = ['heritage', 'monument', 'temple', 'fort', 'palace', 
                            'archaeological', 'historic', 'ancient', 'unesco', 
                            'architecture', 'ruins', 'memorial']
        
        # Must have at least one heritage keyword
        if not any(keyword in content_lower for keyword in heritage_keywords):
            return None
        
        article_data = {
            'title': page.title,
            'url': page.url,
            'content': page.content,
            'summary': page.summary,
            'categories': page.categories,
            'source': 'Indian Heritage (Auto-discovered)',
            'metadata': {
                'region': 'India',
                'heritage_type': 'Indian Monument',
                'discovery_method': 'Wikipedia List Scraping'
            },
            'fetched_at': datetime.now().isoformat()
        }
        
        return article_data
        
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"  ⚠ Disambiguation: {site_name}")
        return None
    except wikipedia.exceptions.PageError:
        print(f"  ⚠ Page not found: {site_name}")
        return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

def main():
    print("="*60)
    print("INDIAN HERITAGE SCRAPER (AUTO-DISCOVERY)")
    print("="*60)
    
    # Step 1: Auto-discover sites from Wikipedia lists
    discovered_sites = scrape_indian_heritage_lists()
    
    if not discovered_sites:
        print("✗ No sites discovered. Exiting.")
        return
    
    # Limit to first 50 for time constraints
    sites_to_fetch = discovered_sites[:50]
    
    print(f"\n[Phase 2] Fetching details for {len(sites_to_fetch)} sites...")
    print("="*60 + "\n")
    
    articles = []
    
    for idx, site_name in enumerate(sites_to_fetch, 1):
        print(f"[{idx}/{len(sites_to_fetch)}] {site_name}")
        
        article = fetch_heritage_article(site_name)
        if article:
            articles.append(article)
            print(f"  ✓ Collected: {article['title']}")
        
        time.sleep(1)  # Rate limiting
    
    # Save articles
    print(f"\n[Saving] {len(articles)} validated heritage documents...")
    os.makedirs('data/raw/indian_heritage', exist_ok=True)
    
    for i, article in enumerate(articles):
        safe_title = "".join(c for c in article['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"indian_{i+1:03d}_{safe_title[:50]}.json"
        filepath = os.path.join('data/raw/indian_heritage', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False)
    
    # Save discovery log
    log_data = {
        'total_discovered': len(discovered_sites),
        'fetched': len(sites_to_fetch),
        'successful': len(articles),
        'sites_list': discovered_sites,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('data/raw/indian_heritage/discovery_log.json', 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("INDIAN HERITAGE COLLECTION COMPLETE")
    print("="*60)
    print(f"Sites discovered: {len(discovered_sites)}")
    print(f"Articles collected: {len(articles)}")
    print(f"Success rate: {len(articles)/len(sites_to_fetch)*100:.1f}%")
    print(f"Saved to: data/raw/indian_heritage/")
    print("="*60)

if __name__ == "__main__":
    main()