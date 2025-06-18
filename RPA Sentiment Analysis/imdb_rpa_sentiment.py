import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from transformers import pipeline

# --- SETUP SELENIUM ---
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)

# --- SCRAPE MOST POPULAR MOVIES ---
url = "https://www.imdb.com/chart/moviemeter/?ref_=nv_mv_mpm"
driver.get(url)
time.sleep(3)

soup = BeautifulSoup(driver.page_source, "html.parser")
movies = []
for row in soup.select("tbody.lister-list tr"):
    title_col = row.find("td", class_="titleColumn")
    if not title_col:
        continue
    title = title_col.a.text.strip()
    year = title_col.span.text.strip("()") if title_col.span else ""
    link = "https://www.imdb.com" + title_col.a["href"].split("?")[0]
    rating_col = row.find("td", class_="imdbRating")
    rating = rating_col.strong.text.strip() if rating_col and rating_col.strong else ""
    movies.append({"title": title, "year": year, "rating": rating, "link": link})

# --- SCRAPE REVIEWS FOR EACH MOVIE ---
def get_reviews(movie_url, max_reviews=10):
    reviews_url = movie_url.rstrip("/") + "/reviews"
    driver.get(reviews_url)
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    review_divs = soup.select(".review-container .text.show-more__control")
    reviews = [div.text.strip() for div in review_divs[:max_reviews]]
    return reviews

# --- LOAD SENTIMENT ANALYSIS PIPELINE ---
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# --- PROCESS EACH MOVIE ---
results = []
for movie in movies[:10]:  # Limit to top 10 for demo; remove slice for all
    print(f"Processing: {movie['title']}")
    reviews = get_reviews(movie["link"])
    sentiments = []
    for review in reviews:
        result = sentiment_analyzer(review[:512])[0]  # Truncate to 512 tokens
        sentiments.append(1 if result["label"] == "POSITIVE" else 0)
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else None
    results.append({
        "title": movie["title"],
        "year": movie["year"],
        "rating": movie["rating"],
        "link": movie["link"],
        "num_reviews": len(reviews),
        "avg_sentiment": avg_sentiment,
    })

# --- OUTPUT RESULTS ---
df = pd.DataFrame(results)
print(df)
df.to_csv("imdb_popular_movies_sentiment.csv", index=False)

driver.quit() 