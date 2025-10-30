import subprocess
import os
from datetime import datetime

def run_scraper(script_name, description):
    """Run a scraper script"""
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"{'='*70}\n")
    
    try:
        subprocess.run(['python', script_name], check=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("MULTI-SOURCE HERITAGE DOCUMENT COLLECTOR")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will collect documents from 4 sources:")
    print("  1. Wikipedia (general heritage)")
    print("  2. UNESCO World Heritage Sites")
    print("  3. Indian Heritage Monuments")
    print("  4. Archive.org Historical Documents")
    print("\nEstimated time: 20-30 minutes")
    print("="*70)
    
    input("\nPress Enter to start collection...")
    
    scrapers = [
        ('src/1a_collect_wikipedia.py', 'Wikipedia Heritage Scraper'),
        ('src/1c_collect_indian_heritage.py', 'Indian Heritage Scraper'),
        ('src/1d_collect_archives.py', 'Archive.org Scraper'),
        ('src/1b_collect_unesco.py', 'UNESCO Scraper (slowest - save for last)'),
    ]
    
    results = []
    for script, desc in scrapers:
        if os.path.exists(script):
            success = run_scraper(script, desc)
            results.append((desc, success))
        else:
            print(f"\n⚠ Warning: {script} not found, skipping...")
            results.append((desc, False))
    
    # Summary
    print("\n" + "="*70)
    print("COLLECTION SUMMARY")
    print("="*70)
    
    for desc, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {desc}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nCheck data/raw/ for collected documents")
    print("="*70)

if __name__ == "__main__":
    main()