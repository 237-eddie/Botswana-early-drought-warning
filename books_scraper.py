import requests
from bs4 import BeautifulSoup
import csv

quotes_data = []
page = 1

while True:
    url = f"http://quotes.toscrape.com/page/{page}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    quote_blocks = soup.find_all("div", class_="quote")
    if not quote_blocks:
        break  # no more quotes, stop looping

    for quote_block in quote_blocks:
        text = quote_block.find("span", class_="text").get_text()
        author = quote_block.find("small", class_="author").get_text()
        quotes_data.append([text, author])

    print(f"Scraped page {page}")
    page += 1

with open("quotes.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Quote", "Author"])
    writer.writerows(quotes_data)

print(f"Saved {len(quotes_data)} quotes total to quotes.csv")