import requests
from bs4 import BeautifulSoup
import json
import os
import time
from datetime import datetime

def scrape_unesco_list():
    """Scrape UNESCO World Heritage Sites list"""
    print("\n" + "="*60)
    print("UNESCO WORLD HERITAGE SCRAPER")
    print("="*60)
    
    base_url = "https://whc.unesco.org/en/list/"
    
    try:
        print("\n[1/3] Fetching UNESCO site list...")
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all heritage site links
        site_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/en/list/' in href and href.count('/') == 3:
                site_id = href.split('/')[-1]
                if site_id.isdigit():
                    site_links.append(f"https://whc.unesco.org/en/list/{site_id}")
        
        # Remove duplicates
        site_links = list(set(site_links))
        print(f"✓ Found {len(site_links)} UNESCO heritage sites")
        
        return site_links[:50]  # Limit to 50 sites for time
        
    except Exception as e:
        print(f"✗ Error fetching UNESCO list: {e}")
        return []

def scrape_unesco_site(url):
    """Scrape individual UNESCO heritage site"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_elem = soup.find('h1')
        title = title_elem.text.strip() if title_elem else "Unknown"
        
        # Extract description
        description = ""
        desc_divs = soup.find_all('div', class_='description')
        for div in desc_divs:
            description += div.get_text(strip=True, separator=' ') + " "
        
        # Extract metadata
        metadata = {}
        
        # Find country
        country_elem = soup.find('a', href=lambda x: x and '/en/states-parties/' in x)
        if country_elem:
            metadata['country'] = country_elem.text.strip()
        
        # Find category (Cultural/Natural/Mixed)
        category_elem = soup.find('div', class_='category')
        if category_elem:
            metadata['category'] = category_elem.text.strip()
        
        # Find criteria
        criteria = []
        criteria_elems = soup.find_all('li', class_='criterion')
        for crit in criteria_elems:
            criteria.append(crit.text.strip())
        metadata['criteria'] = criteria
        
        # Find year inscribed
        year_elem = soup.find(text=lambda x: x and 'Date of Inscription:' in str(x))
        if year_elem:
            year_text = year_elem.strip().split(':')[-1].strip()
            metadata['year_inscribed'] = year_text
        
        article_data = {
            'title': title,
            'url': url,
            'content': description,
            'summary': description[:500] + "..." if len(description) > 500 else description,
            'source': 'UNESCO World Heritage',
            'metadata': metadata,
            'fetched_at': datetime.now().isoformat()
        }
        
        return article_data
        
    except Exception as e:
        print(f"✗ Error scraping {url}: {e}")
        return None

def main():
    # Get list of UNESCO sites
    site_urls = scrape_unesco_list()
    
    if not site_urls:
        print("✗ No UNESCO sites found. Exiting.")
        return
    
    print(f"\n[2/3] Scraping {len(site_urls)} UNESCO sites...")
    print("This will take ~10-15 minutes...\n")
    
    articles = []
    for idx, url in enumerate(site_urls, 1):
        print(f"[{idx}/{len(site_urls)}] Scraping: {url}")
        
        article = scrape_unesco_site(url)
        if article:
            articles.append(article)
            print(f"✓ {article['title']}")
        
        time.sleep(2)  # Be respectful to UNESCO servers
    
    # Save articles
    print(f"\n[3/3] Saving {len(articles)} articles...")
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
    print(f"Total documents: {len(articles)}")
    print(f"Saved to: data/raw/unesco/")

if __name__ == "__main__":
    main()