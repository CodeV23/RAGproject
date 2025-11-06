"""
Wikipedia Scraper
------------------------------------
Fetches Wikipedia articles by topic name
and saves them as plain text files in data/raw/.
"""

import os
import re
import wikipediaapi

def sanitize_filename(name: str) -> str:
    """Clean up topic names for safe filenames."""
    return re.sub(r'[^a-zA-Z0-9_-]+', '_', name.strip().lower())

def scrape_wikipedia_article(topic: str, out_dir="data/raw"):
    """Fetch and save a Wikipedia article as .txt"""
    os.makedirs(out_dir, exist_ok=True)
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='Vedant-CS599-FinalProject (https://github.com/CodeV23)'
    )

    page = wiki.page(topic)

    if not page.exists():
        print(f"[ERROR] No Wikipedia page found for '{topic}'.")
        return

    # Save text
    filename = sanitize_filename(topic) + ".txt"
    out_path = os.path.join(out_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {page.title}\n\n{page.text}")

    print(f"[OK] Saved article '{page.title}' → {out_path}")

def main():
    topic = input("Enter a topic to scrape: ")
    scrape_wikipedia_article(topic)

if __name__ == "__main__":
    main()
