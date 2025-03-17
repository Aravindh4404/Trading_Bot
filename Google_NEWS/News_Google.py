import requests
from bs4 import BeautifulSoup
import json
import time


def fetch_google_search_news(topic):
    base_url = "https://www.google.com/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    news_list = []
    page = 0
    max_pages = 10  # Arbitrary limit to avoid infinite loop; adjust as needed

    while page < max_pages:
        params = {
            "q": topic,
            "tbm": "nws",  # News filter
            "tbs": "qdr:d",  # Last 24 hours
            "hl": "en",
            "gl": "us",
            "start": page * 10  # Pagination: 10 results per page
        }

        try:
            print(f"Sending request to Google for page {page + 1}...")
            response = requests.get(base_url, params=params, headers=headers)
            print(f"Response status code: {response.status_code}")

            if response.status_code != 200:
                print(f"Failed to fetch page {page + 1}: {response.status_code}")
                break

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find news articles
            articles = soup.find_all('div', class_='SoaBEf')

            if not articles:
                print(f"No articles found with class 'SoaBEf' on page {page + 1}. Trying broader search...")
                articles = soup.find_all('div', class_='g')

            if not articles:
                print(f"No articles found on page {page + 1}. Stopping pagination.")
                break

            print(f"Found {len(articles)} potential articles on page {page + 1}.")

            articles_processed = False
            for article in articles:
                # Extract headline
                headline_tag = article.find('div', role='heading') or article.find('h3')
                headline = headline_tag.text.strip() if headline_tag else "No headline"
                print(f"Headline found: {headline}")

                # Extract link
                link_tag = article.find('a', href=True)
                link = link_tag['href'] if link_tag and link_tag['href'].startswith('http') else "No link"
                print(f"Link found: {link}")

                # Extract time
                time_str = "Unknown time"
                for span in article.find_all('span'):
                    time_text = span.text.strip()
                    print(f"Time text candidate: '{time_text}'")
                    if "ago" in time_text.lower():
                        if "min" in time_text.lower():
                            minutes_ago = int(''.join(filter(str.isdigit, time_text)))
                            time_str = f"{minutes_ago} mins ago"
                        elif "hour" in time_text.lower():
                            hours_ago = int(''.join(filter(str.isdigit, time_text)))
                            time_str = f"{hours_ago} hours ago"
                        break
                    elif time_text.lower() in ["business insider", "cleantechnica", "mint"]:
                        continue

                print(f"Processed time: {time_str}")

                # Since tbs=qdr:d ensures 24-hour limit, add all articles
                news_item = {
                    "headline": headline,
                    "link": link,
                    "time": time_str
                }
                news_list.append(news_item)
                articles_processed = True

            # If no articles were processed (e.g., time not found), assume end of relevant results
            if not articles_processed:
                print(f"No valid articles processed on page {page + 1}. Stopping.")
                break

            # Delay to avoid rate-limiting
            time.sleep(1)
            page += 1

        except Exception as e:
            print(f"Error fetching news on page {page + 1}: {str(e)}")
            break

    return news_list


def save_to_json(news_items, filename="news_data.json"):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(news_items, f, indent=4, ensure_ascii=False)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving to JSON: {e}")


def main():
    topic = input("Enter the news topic you want to search for: ")
    print(f"Fetching latest news on '{topic}' from the past 24 hours across multiple pages...\n")

    news_items = fetch_google_search_news(topic)

    if news_items:
        for idx, item in enumerate(news_items, 1):
            print(f"News {idx}:")
            print(f"headline: {item['headline']}")
            print(f"link: {item['link']}")
            print(f"time: {item['time']}\n")

        save_to_json(news_items)
        print(f"Total news items collected: {len(news_items)}")
    else:
        print("No news found or an error occurred.")


if __name__ == "__main__":
    main()
