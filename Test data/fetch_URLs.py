import requests
from bs4 import BeautifulSoup

# Function to fetch the latest news article URLs from BBC News
def fetch_latest_bbc_news_urls():
    url = "https://www.bbc.com/news"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch the URL. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    # Debugging: Print the first 2000 characters of the HTML content
    print(soup.prettify()[:2000])

    # Find all article links
    articles = soup.find_all('a', href=True)
    print(f"Found {len(articles)} links in total")

    # Extract and filter URLs
    article_urls = []
    for article in articles:
        href = article.get('href')
        # Check if the link is a news article and not a topic page
        if href and '/news/' in href and not href.endswith('/news') and '/news/live' not in href:
            # Check for article pattern (e.g., URLs containing digits)
            if any(char.isdigit() for char in href):
                full_url = f"https://www.bbc.com{href}" if href.startswith('/') else href
                if full_url not in article_urls:
                    article_urls.append(full_url)
                    print(f"Article URL: {full_url}")  # Debugging: Print each found article URL
        if len(article_urls) >= 20:
            break

    return article_urls

# Function to write the URLs to a file
def write_urls_to_file(urls, filename):
    with open(filename, 'w') as file:
        for url in urls:
            file.write(url + '\n')

# Main script to fetch and store the URLs
if __name__ == "__main__":
    urls = fetch_latest_bbc_news_urls()
    if urls:
        write_urls_to_file(urls, 'Test Data/test-urls.txt')
        print("The latest 10 news article URLs from BBC News have been stored in 'test-urls.txt'.")
    else:
        print("No URLs found or an error occurred.")
