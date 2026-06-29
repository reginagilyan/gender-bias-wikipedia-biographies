# %%
# load necessary packages
import wikipediaapi
import requests
from bs4 import BeautifulSoup
import json
import time
import aiohttp
import asyncio
import nest_asyncio
import pandas as pd
from mw.api import Session
from mw.lib import reverts
import urllib.parse
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
from tqdm import tqdm
from tqdm.asyncio import tqdm
from scipy.stats import ttest_ind
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora
from gensim.models import LdaModel
from collections import Counter
from wordcloud import WordCloud
# %%
# %%
# set up user agent for Wikipedia API
headers = {'User-Agent': "WikipediaBiasResearch/1.0 (regina.gilyan@uni-konstanz.de)"}

# Wikipedia URL of Vital Articles Level 4 - People
vital_articles_url = "https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/4/People"

# %%
### EXTRACTING THE CONTENT OF THE VITAL ARTICLES PAGE

response = requests.get(vital_articles_url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# creating an empty list to store the data
article_data = {}


# upon inspecting the HTML structure, we can extract article titles, 
#   relative URLs as well as quality of the article at the same time

for li in soup.select("li"):  # Select all <li> elements
    # Extract article title and link
    a_tags = li.find_all("a", href=True, title=True)
    if a_tags:
        page_link = None
        page_title = None
        rating = None
        
        for a_tag in a_tags:
            if a_tag["href"].startswith("/wiki/") and not ":" in a_tag["href"]:
                page_link = a_tag["href"]
                page_title = a_tag.text.strip()
                break  # Stop after finding the first valid article link
        
        # extracting article quality
        span = li.find("span", class_="noviewer")
        if span:
            rating_tag = span.find("a") or span.find("span")  # Check both <a> and <span>
            if rating_tag:
                rating = rating_tag.get("title")
        
        # Store data in dictionary
        if page_title and page_link:
            article_data[page_title] = {"link": page_link, "rating": rating}


# print results to see if it worked
for key, value in list(article_data.items())[:20]:
    print(f"{key}: {value}")

# %%
# print pages where the rating is None, if there's any
pages_with_no_rating = {title: data for title, data in article_data.items() if data["rating"] is None}

print(f"Number of pages with no rating: {len(pages_with_no_rating)}")
for key, value in list(pages_with_no_rating.items()):
    print(f"{key}: {value}")
# %%
# detected that we got the Wikipedia "Main page" as well, so deleting that
keys_to_remove = [key for key in article_data.keys() if key.lower() == "main page"]
for key in keys_to_remove:
    del article_data[key]

# print the total number of pages we got to check if it's 1998
print(f"Total number of pages extracted: {len(article_data)}")

# %%
### EXTRACT WIKIDATA IDS
# we will need this to extract the gender of the person that the page is about

nest_asyncio.apply()

async def fetch_wikidata_id(session, link):
    """Fetch Wikidata ID for a Wikipedia page using the link."""
    # Extract the page title from the link (remove '/wiki/' part)
    title = link.lstrip('/wiki/')
    
    url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={title}&prop=pageprops&ppprop=wikibase_item"
    
    async with session.get(url) as response:
        data = await response.json()
        pages = data.get("query", {}).get("pages", {})
        
        for page in pages.values():
            return page.get("pageprops", {}).get("wikibase_item", "None")
    
    return "None"

# Function to fetch Wikidata IDs for all articles in article_data
async def get_wikidata_ids(article_data):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_wikidata_id(session, title) for title in article_data.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for (title, wikidata_id) in zip(article_data.keys(), results):
        if isinstance(wikidata_id, Exception):  
            print(f"Error fetching {title}: {wikidata_id}")
            article_data[title]["wikidata_id"] = "None"
        else:
            article_data[title]["wikidata_id"] = wikidata_id if wikidata_id else "None"

# run function and update dictionary
asyncio.run(get_wikidata_ids(article_data))

# print to verify
for key, value in list(article_data.items())[:10]:
    print(f"{key}: {value}")

# %%
# checking if there are articles that didn't return a wikidata id
none_wikidata_ids = [title for title, data in article_data.items() if data["wikidata_id"] == "None"]

if none_wikidata_ids:
    print(f"The following pages have no Wikidata ID: {none_wikidata_ids}")
else:
    print("All pages have a valid Wikidata ID.")

# %%
# unfortunately one page ('Abu al-Qasim al-Zahrawi') returned no id
# I checked on wikidata and there's an entry so it should have an id
# as a next step, I'm going to check "manually" if our relative link is correct

if 'Abu al-Qasim al-Zahrawi' in article_data:
    print(f"Data for Abu al-Qasim al-Zahrawi: {article_data['Abu al-Qasim al-Zahrawi']}")
else:
    print("No entry found for Abu al-Qasim al-Zahrawi.")

# the relative link is correct... 
# I tried a few debugging steps, but idk where it went wrong for this one page
# %%
# I'm going to manually put it's wikidata id in the dictionary
article_data["Abu al-Qasim al-Zahrawi"]["wikidata_id"] = "Q311495"

# %%
# removing '/wiki/' prefix from link values to make life easier
for key, entry in article_data.items():
    if entry["link"].startswith("/wiki/"):
        entry["link"] = entry["link"].lstrip("/wiki/") 
# %%
### EXTRACT AND CLEAN FULL TEXTS

# function to fetch full text using mediawiki API
async def get_wikipedia_text(session, title):
    url = f"https://en.wikipedia.org/w/api.php?action=parse&page={title}&format=json&prop=text"
    
    async with session.get(url) as response:
        data = await response.json()
        
        if "parse" in data and "text" in data["parse"]:
            return data["parse"]["text"]["*"]  # extracting full HTML content
        else:
            return None  # in case there's an error

# Function to clean HTML
def clean_wikipedia_text(html_text):
    if html_text is None:
        return None  # skip if no text is available (shouldn't happen)
    
    soup = BeautifulSoup(html_text, "html.parser")
    
    # Remove tables, references, and unnecessary content
    for tag in soup.find_all(["table", "sup", "style", "script"]):
        tag.decompose()

    # Extract plain text
    clean_text = soup.get_text(separator=" ", strip=True)
    
    return clean_text

# Function to retrieve and clean texts for all pages (without removing /wiki/ prefix)
async def get_clean_texts_from_dict(article_data):
    async with aiohttp.ClientSession() as session:
        # Create tasks to fetch full text for each page using the cleaned 'link'
        tasks = [get_wikipedia_text(session, entry["link"]) for entry in article_data.values()]
        raw_texts = await asyncio.gather(*tasks)
        
        # Clean the extracted texts and store them back in article_data
        for idx, (key, entry) in enumerate(article_data.items()):
            raw_text = raw_texts[idx]
            cleaned_text = clean_wikipedia_text(raw_text)
            entry["full_text"] = cleaned_text  # Store cleaned text in the corresponding entry
        
    return article_data

# run function
article_data = asyncio.run(get_clean_texts_from_dict(article_data))

# verify result: check for pages where full_text is None
pages_without_full_text = [title for title, data in article_data.items() if data.get('full_text') is None]

if pages_without_full_text:
    print(f"The following pages didn't return full text: {', '.join(pages_without_full_text)}")
else:
    print("All pages returned full text successfully.")

# got all texts :)
# %%
### EXTRACT NUMBER OF SECTIONS IN EACH ARTICLE

# function to count sections from html
def count_sections_from_html(html_text):
    if html_text is None:
        return 0
    soup = BeautifulSoup(html_text, "html.parser")
    # wikipedia sections in the parsed HTML are inside <h2>, <h3>, <h4>, etc.
    headings = soup.find_all(re.compile('^h[2-6]$'))
    return len(headings)

# function to fetch and count sections for each page using 'link' values from dict
async def get_sections_count(article_data):
    async with aiohttp.ClientSession() as session:
        tasks = [get_wikipedia_text(session, entry["link"]) for entry in article_data.values()]
        raw_htmls = await asyncio.gather(*tasks)

        # store section counts in dict
        for idx, (key, entry) in enumerate(article_data.items()):
            entry["num_sections"] = count_sections_from_html(raw_htmls[idx])
    
    return article_data

# run function
article_data = asyncio.run(get_sections_count(article_data))

# verify result
print(f"Total articles processed: {len(article_data)}")
for key, entry in article_data.items():
    print(f"{key}: {entry['num_sections']} sections")

# %%
### EXTRACT GENDER

# function to fetch gender information for a given Wikidata ID
async def get_gender(session, wikidata_id):
    url = f"https://www.wikidata.org/w/api.php?action=wbgetclaims&format=json&entity={wikidata_id}"
    
    async with session.get(url) as response:
        data = await response.json()
        claims = data.get("claims", {})
        
        # Check for gender (P21) claim
        if "P21" in claims:
            for claim in claims["P21"]:
                gender_id = claim["mainsnak"]["datavalue"]["value"]["id"]
                if gender_id == "Q6581097": 
                    return "male"
                elif gender_id == "Q6581072":
                    return "female"
                else:
                    return "other"  # unknown or other gender

    return "unknown"  # in case no gender is found

# Function to fetch gender information for all Wikidata IDs and update the dictionary
async def get_genders_for_dict(article_data):
    async with aiohttp.ClientSession() as session:
        tasks = [get_gender(session, entry["wikidata_id"]) for entry in article_data.values()]
        genders = await asyncio.gather(*tasks)
    
        # Store gender info in the dictionary
        for (key, entry), gender in zip(article_data.items(), genders):
            entry["gender"] = gender
    
    return article_data


# run function
article_data = asyncio.run(get_genders_for_dict(article_data))

# verify result
for key, entry in article_data.items():
    print(f"{key}: {entry['gender']}")

# %%
### EXTRACT HYPERLINKS

# function to extract hyperlinks from a given Wikipedia page URL
# added a lot of debugging steps because I kept getting different errors on the first few tries of running this
async def get_hyperlinks(session, url, retries=3):
    for attempt in range(retries):
        try:
            timeout = aiohttp.ClientTimeout(total=30)  # Increase timeout
            async with session.get(url, timeout=timeout) as response:
                if response.status != 200:
                    print(f"⚠️ Warning: {url} returned status {response.status}")
                    return []  # skip this page

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                links = []
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if href.startswith('/wiki/') and not any(sub in href for sub in [":", "#"]):
                        full_url = f"https://en.wikipedia.org{href}"
                        links.append(full_url)

                return links  # Success!

        except Exception as e:
            print(f"❌ Error fetching {url} (Attempt {attempt+1}/{retries}): {e}")
            await asyncio.sleep(2)  # small delay before retrying

    print(f"🚨 Failed to fetch {url} after {retries} attempts.")
    return []  # give up after retries

# function to extract hyperlinks for all pages using 'link' values from the dictionary
async def extract_all_hyperlinks(article_data):
    async with aiohttp.ClientSession() as session:
        for title, data in article_data.items():
            if "link" not in data:
                print(f"❌ Skipping {title} - No 'link' found")
                data["hyperlinks"] = []
                continue
            
            url = f"https://en.wikipedia.org/wiki/{data['link']}"  # construct full URL

            print(f"🔍 Fetching hyperlinks for {title}: {url}")  # debugging print

            try:
                links = await get_hyperlinks(session, url)
                data["hyperlinks"] = links
                print(f"✅ {title}: {len(links)} links extracted")
            except Exception as e:
                print(f"❌ Error fetching {title}: {e}")
                data["hyperlinks"] = []

            await asyncio.sleep(1)  # prevent rate-limiting
# run function
asyncio.run(extract_all_hyperlinks(article_data))

# %%
# removing main page links, as those are irrelevant 
# define full URL of the Main Page
main_page_link = "https://en.wikipedia.org/wiki/Main_Page"

# loop through article_data and remove
for data in article_data.values():
    if "hyperlinks" in data:
        data["hyperlinks"] = [link for link in data["hyperlinks"] if link != main_page_link]

print("Main Page links removed from article_data.")

# %%
# start saving dict just in case
with open("article_data_dict.json", "w", encoding="utf-8") as f:
    json.dump(article_data, f, ensure_ascii=False, indent=4)

# %%
### EXTRACT REVISION HISTORY
# going to start storing this in a seperate dict, as the other one is getting big 

async def fetch_revisions_for_page(session, title):
    """Fetch all revisions with pagination for one Wikipedia page."""
    revisions = []
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvlimit": "500",
        "rvprop": "ids|timestamp|user|userid|sha1",
        "format": "json",
        "formatversion": "2"
    }

    while True:
        async with session.get("https://en.wikipedia.org/w/api.php", params=params) as response:
            data = await response.json()
            pages = data.get('query', {}).get('pages', [])
            if pages and 'revisions' in pages[0]:
                revisions.extend(pages[0]['revisions'])
            cont = data.get('continue', {}).get('rvcontinue')
            if cont:
                params['rvcontinue'] = cont
            else:
                break
    return revisions

def revert_list(revs):
    """Detect reverts based on SHA1 hash values."""
    rev_events = ((rev['sha1'], rev) for rev in revs if 'sha1' in rev.keys())
    detected_reverts = []
    for revert in reverts.detect(rev_events):
        detected_reverts.append((revert.reverting['revid'], revert.reverted_to['revid']))
    return detected_reverts

async def fetch_revisions_and_reverts_for_page(session, title, full_url):
    revisions = await fetch_revisions_for_page(session, title)
    rev_reverts = revert_list(revisions)
    return title, {"revisions": revisions, "reverts": rev_reverts}

async def fetch_all_revisions(article_data, max_concurrent=5):
    """Fetch revisions for all pages concurrently with a progress bar."""
    semaphore = asyncio.Semaphore(max_concurrent)
    revisions_data = {}  # Separate dictionary for revisions

    async with aiohttp.ClientSession() as session:
        async def sem_fetch(title, full_url):
            async with semaphore:
                return await fetch_revisions_and_reverts_for_page(session, title, full_url)

        tasks = []
        for title, data in article_data.items():
            if "link" in data:
                full_url = f"https://en.wikipedia.org/wiki/{data['link']}"
                tasks.append(sem_fetch(title, full_url))
            else:
                print(f"⚠️ Skipping {title} (No 'link' found)")

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching revisions"):
            try:
                title, rev_data = await future
                revisions_data[title] = rev_data
            except Exception as e:
                print(f"❌ Error fetching {title}: {e}")

    return revisions_data


# fetch revisions and save them separately
revisions_data = asyncio.run(fetch_all_revisions(article_data, max_concurrent=5))

# %%
# save revisions to a new JSON file
with open("revisions_dict.json", "w", encoding="utf-8") as f:
    json.dump(revisions_data, f, ensure_ascii=False, indent=4)

# %%
# EXTRACT NUMBER OF UNIQUE EDITORS

# Function to get the unique editor count for a single Wikipedia page
async def get_unique_editors_count(session, link):
    base_url = "https://en.wikipedia.org/w/api.php"
    full_url = f"https://en.wikipedia.org/wiki/{link}"  # Construct full URL

    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": link,
        "rvprop": "user",  # Only fetch editor usernames
        "rvlimit": "max",  # Get as many revisions as allowed
    }

    editors = set()  # Store unique editor names

    while True:
        try:
            async with session.get(base_url, params=params) as response:
                if response.status == 429:  # Handle rate limits
                    print(f"Rate limited on {link}, waiting 5 seconds...")
                    await asyncio.sleep(5)
                    continue

                data = await response.json()
                pages = data.get("query", {}).get("pages", {})
                
                for page_id, page_data in pages.items():
                    revisions = page_data.get("revisions", [])
                    for revision in revisions:
                        user = revision.get("user")
                        if user:
                            editors.add(user)  # Add unique editors

                if "continue" not in data:  # Stop if no more revisions
                    break
                params.update(data["continue"])

            await asyncio.sleep(1)  # Small delay to avoid hitting API limits

        except Exception as e:
            print(f"Error fetching editors for {link}: {e}")
            break

    return link, len(editors)  # Return link and unique editor count

# function to fetch editor counts for all pages and store them in revisions_data
async def fetch_all_editors_counts(article_data, revisions_data):
    async with aiohttp.ClientSession() as session:
        tasks = {link: get_unique_editors_count(session, link) for link in article_data.keys()}
        results = await asyncio.gather(*tasks.values())

    # Store results in revisions_data
    for link, editor_count in results:
        if link in revisions_data:
            revisions_data[link]["num_editors"] = editor_count
        else:
            revisions_data[link] = {"num_editors": editor_count}  # Initialize if missing

    return revisions_data


# run function
revisions_data = asyncio.run(fetch_all_editors_counts(article_data, revisions_data))

# save updated revisions_data
with open("revisions_dict.json", "w") as f:
    json.dump(revisions_data, f, indent=4)

print("Editor counts added to revisions_data!")

# %%
### PREPARE FOR ANALYSIS BY CREATING THEMATIC DFs

# df for general stats/ descriptive analysis
# extract relevant data from article_data
data_list = []
for title, data in article_data.items():
    data_list.append({
        "title": title,
        "article_quality": data.get("rating", "Unknown"),
        "num_words": len(data.get("full_text", "").split()),  # Word count
        "num_sections": data.get("num_sections", 0),  # Default to 0 if missing
        "gender": data.get("gender", "Unknown"),
        "num_hyperlinks": len(data.get("hyperlinks", [])),  # Count hyperlinks
    })

# convert to df
article_df = pd.DataFrame(data_list)

# df for revisions data
revisions_df = pd.DataFrame([
    {
        "page_title": page,
        "num_revisions": len(data["revisions"]),
        "num_reverts": len(data["reverts"]),
        "num_editors": data["num_editors"],  # Directly added
    }
    for page, data in revisions_data.items()
])

# add gender to the df too:
# extract mapping from article_data
title_to_gender = {title: data.get("gender", "Unknown") for title, data in article_data.items()}

# map the gender values into df
revisions_df["gender"] = revisions_df["page_title"].map(title_to_gender)

# df for text (sentiment analysis)
text_data = []
for title, data in article_data.items():
    text_data.append({
        "title": title,
        "gender": data.get("gender", "Unknown"),  
        "full_text": data.get("full_text", ""), 
    })

text_df = pd.DataFrame(text_data)

# %%
### SAVE
article_df.to_csv('final_articles.csv', index=False)
revisions_df.to_csv('final_revisions.csv', index=False)
text_df.to_csv('final_text.csv', index=False)

# %%
### PREPROCESSING/ DATA CLEANING

# check gender distribution
gender_counts = article_df['gender'].value_counts(dropna=False)
print(gender_counts)

gender_percent = article_df['gender'].value_counts(normalize=True) * 100
print(gender_percent)

# checking where gender is unknown
unknown_pages = article_df[article_df['gender'] == 'unknown']
unknown_pages

### REMOVE GENDER = 'UNKNOWN'
# deleting unknown genders, as they are not truly missings or anything but rather groups/ bands/ families etc. 

# get a list of irrelevant titles
irrelevant_titles = unknown_pages['title'].tolist()

    ## REMOVE FROM ALL DFS

# drop from article_df
article_df = article_df[~article_df['title'].isin(irrelevant_titles)].reset_index(drop=True)
# drop from revisions_df
revisions_df = revisions_df[~revisions_df['page_title'].isin(irrelevant_titles)].reset_index(drop=True)
# drop from text_df
text_df = text_df[~text_df['title'].isin(irrelevant_titles)].reset_index(drop=True)

# check length, verify it worked
print(len(article_df))
print(len(revisions_df))
print(len(text_df))
# all 3 dfs now have 1971 rows

### RECALCULATE GENDER COUNTS & PERCENT

gender_counts = article_df['gender'].value_counts(dropna=False)
print(gender_counts)

gender_percent = article_df['gender'].value_counts(normalize=True) * 100
print(gender_percent)

# plot
plt.figure(figsize=(6, 4))
sns.countplot(data=article_df, x="gender", palette=["#ff9999", "#66b3ff"])
plt.title("Number of Biographies by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# %%
### CLEAN FULL TEXT
# ... or clean it more ...
# ... by removing sections of references, notes etc. from full text


def clean_biography(text):
    # Remove sections starting from 'References' or 'Notes'
    text = re.sub(r'(?i)(References|Notes|Publications|Works Cited|Further Reading|External Links).*', '', text, flags=re.DOTALL)
    return text

text_df['plain_content'] = text_df['full_text'].apply(clean_biography)
# cleaned in text_df

    ### COUNT WORDS IN PLAIN TEXT 

# function to count words in text
def count_words(text):
    return len(text.split())

# add a new column with the word count of 'plain_content'
text_df['plain_word_count'] = text_df['plain_content'].apply(count_words)

# add the plain_word_count to article_df as well for descriptive analysis
article_df = article_df.merge(text_df[['title', 'plain_word_count']], on='title', how='left')
# %%
### DESCRIPTIVE STATISTICS: TEXT & PAGE CHARACTERISTICS

# summary stats by gender
print(article_df.groupby("gender")[["plain_word_count", "num_sections", "num_hyperlinks"]].agg(
    ["mean", "median", "min", "max", "std"]))

# there's nothing really interesting here, so I'll just note the mean & SD for my report
print(article_df.groupby("gender")[["plain_word_count", "num_sections", "num_hyperlinks"]].agg(["mean", "std"]))

# %%
# PLOT DENSITIES OF NUMERIC VARS: TEXT & PAGE CHARACTERISTICS

sns.set_style("whitegrid")

# define colors
colors = ["#c2a5cf", "#fdae61"]

# figure with 3 subplots in a row
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# main title
plt.suptitle("Page characteristics: density plots", fontsize=16, fontweight="bold")

# Word count density
sns.kdeplot(data=article_df, x="plain_word_count", hue="gender", fill=True, common_norm=False, alpha=0.5, palette=colors, ax=axes[0])
axes[0].set_title("Words", fontsize=12)
axes[0].set_xlabel("Number of words")
axes[0].set_ylabel("Density")
axes[0].set_xlim(0, None)

# Number of sections
sns.kdeplot(data=article_df, x="num_sections", hue="gender", fill=True, common_norm=False, alpha=0.5, palette=colors, ax=axes[1])
axes[1].set_title("Sections", fontsize=12)
axes[1].set_xlabel("Sections")
axes[1].set_ylabel("")  # Remove y-axis label
axes[1].set_xlim(0, None)

# Number of hyperlinks
sns.kdeplot(data=article_df, x="num_hyperlinks", hue="gender", fill=True, common_norm=False, alpha=0.5, palette=colors, ax=axes[2])
axes[2].set_title("Hyperlinks", fontsize=12)
axes[2].set_xlabel("Hyperlinks")
axes[2].set_ylabel("")  # Remove y-axis label
axes[2].set_xlim(0, None)

# Adjust grid line transparency
for ax in axes:
    ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit main title
plt.show()

# %%
### BOXPLOTS

sns.set_style("whitegrid")

# Create figure with 3 subplots in a row
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

# Set main title
plt.suptitle("Page characteristics: boxplots", fontsize=16, fontweight="bold")

# Word count boxplot
boxplot1 = sns.boxplot(data=article_df, x="gender", y="plain_word_count", palette=colors, ax=axes[0])
axes[0].set_title("Words", fontsize=12)
axes[0].set_xlabel("")
axes[0].set_ylabel("Number of words")
for patch in boxplot1.artists:  # Set transparency for box color
    patch.set_alpha(0.5)

# Number of sections boxplot
boxplot2 = sns.boxplot(data=article_df, x="gender", y="num_sections", palette=colors, ax=axes[1])
axes[1].set_title("Sections", fontsize=12)
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("Number of sections")  
for patch in boxplot2.artists:
    patch.set_alpha(0.5)

# Number of hyperlinks boxplot
boxplot3 = sns.boxplot(data=article_df, x="gender", y="num_hyperlinks", palette=colors, ax=axes[2])
axes[2].set_title("Hyperlinks", fontsize=12)
axes[2].set_xlabel("")
axes[2].set_ylabel("Number of hyperlinks")  
for patch in boxplot3.artists:
    patch.set_alpha(0.5)

# Adjust grid line transparency
for ax in axes:
    ax.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Leaves space at bottom for the legend
plt.show()

# %%
### CHECK WITHOUT OUTLIERS

# calculate the 99th percentile for each variable
upper_limit_word_count = article_df['plain_word_count'].quantile(0.99)
upper_limit_sections = article_df['num_sections'].quantile(0.99)
upper_limit_hyperlinks = article_df['num_hyperlinks'].quantile(0.99)

# filter out values above the 99th percentile
filtered_word_count = article_df[article_df['plain_word_count'] <= upper_limit_word_count]
filtered_sections = article_df[article_df['num_sections'] <= upper_limit_sections]
filtered_hyperlinks = article_df[article_df['num_hyperlinks'] <= upper_limit_hyperlinks]

# print updated means and SDs
print("Word Count:")
print(filtered_word_count.groupby("gender")["plain_word_count"].agg(['mean', 'std']))

print("\nSections:")
print(filtered_sections.groupby("gender")["num_sections"].agg(['mean', 'std']))

print("\nHyperlinks:")
print(filtered_hyperlinks.groupby("gender")["num_hyperlinks"].agg(['mean', 'std']))


# %%
### STATISTICAL TEST FOR WORD COUNT, SECTIONS AND HYPERLINKS

t_stat, p_value = ttest_ind(article_df[article_df["gender"] == "male"]["plain_word_count"], 
                            article_df[article_df["gender"] == "female"]["plain_word_count"], equal_var=False)
print(f"Words T-test: t={t_stat:.3f}, p={p_value:.3f}")
# t=-0.394, p=0.69

t_stat, p_value = ttest_ind(article_df[article_df["gender"] == "male"]["num_sections"], 
                            article_df[article_df["gender"] == "female"]["num_sections"], equal_var=False)
print(f"Sections T-test: t={t_stat:.3f}, p={p_value:.3f}")
# t=0.265, p=0.791

t_stat, p_value = ttest_ind(article_df[article_df["gender"] == "male"]["num_hyperlinks"], 
                            article_df[article_df["gender"] == "female"]["num_hyperlinks"], equal_var=False)
print(f"Hyperlinks T-test: t={t_stat:.3f}, p={p_value:.3f}")
# t=-1.803, p=0.073

### STATISTICAL TESTS FOR FILTERED DFS

t_stat, p_value = ttest_ind(filtered_word_count[filtered_word_count["gender"] == "male"]["plain_word_count"], 
                            filtered_word_count[filtered_word_count["gender"] == "female"]["plain_word_count"], equal_var=False)
print(f"Filtered word count T-test: t={t_stat:.3f}, p={p_value:.3f}")
# Example output: t=-1.893, p=0.059

t_stat, p_value = ttest_ind(filtered_sections[filtered_sections["gender"] == "male"]["num_sections"], 
                            filtered_sections[filtered_sections["gender"] == "female"]["num_sections"], equal_var=False)
print(f"Filtered sections T-test: t={t_stat:.3f}, p={p_value:.3f}")
# Example output: t=-1.234, p=0.217

t_stat, p_value = ttest_ind(filtered_hyperlinks[filtered_hyperlinks["gender"] == "male"]["num_hyperlinks"], 
                            filtered_hyperlinks[filtered_hyperlinks["gender"] == "female"]["num_hyperlinks"], equal_var=False)
print(f"Filtered hyperlinks T-test: t={t_stat:.3f}, p={p_value:.3f}")
# Example output: t=-2.650, p=0.008

# %%
### ARTICLE QUALITY

# compute percentage of articles in each quality level per gender
gender_quality_percent = (
    article_df.groupby(["article_quality", "gender"]).size().unstack()
    .div(gender_counts, axis=1) * 100  # Normalize by gender count
)

# order of article quality categories (from worst to best)
quality_order = ["Start-Class article", "C-Class article", "B-Class article", "Good article", "Featured article"]

# reorder df based on this order
gender_quality_percent = gender_quality_percent.reindex(quality_order)
# remove the word "article" from category labels
gender_quality_percent.index = gender_quality_percent.index.str.replace(" article", "", regex=False)

print(gender_quality_percent)

# PLOT
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 6))
gender_quality_percent.plot(kind="bar", stacked=False, color=colors, alpha=0.6)

# titles and labels
plt.title("Article quality distribution by gender", fontsize=14)
plt.xlabel("Article quality", fontsize=11)
plt.ylabel("Percentage (%)", fontsize=11)
plt.xticks(rotation=45, fontsize=9)
plt.yticks(fontsize=9)
plt.legend(title="gender", title_fontsize=11, fontsize=9)

# grid
plt.grid(axis="y", linestyle="dashed", alpha=0.3)

plt.show()

# %%
### REVISIONS

revisions_df = pd.read_csv('final_revisions.csv')

# add a new column to represent the revert rate as a proportion of total revisions per page
revisions_df['revert_rate'] = revisions_df['num_reverts'] / revisions_df['num_revisions']

# print means and SDs
print("Revision metrics:")
print(revisions_df.groupby("gender")["num_revisions"].agg(['mean', 'std']))

print("\nRevert metrics:")
print(revisions_df.groupby("gender")["num_reverts"].agg(['mean', 'std']))

print("\nEditor metrics:")
print(revisions_df.groupby("gender")["num_editors"].agg(['mean', 'std']))

print("\nRevert rate metrics:")
print(revisions_df.groupby("gender")["revert_rate"].agg(['mean', 'std']))

# %%
### EDITORIAL ACTIVITY BOXPLOTS

sns.set_style("whitegrid")

# Create figure with 3 subplots in a row
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

# Set main title
plt.suptitle("Distribution of editorial activity", fontsize=16, fontweight="bold", y=0.95)

# revisions
boxplot1 = sns.boxplot(data=revisions_df, x="gender", y="num_revisions", palette=colors, ax=axes[0])
axes[0].set_title("Revisions", fontsize=12)
axes[0].set_xlabel("")
axes[0].set_ylabel("Count")
for patch in boxplot1.artists:  # Set transparency for box color
    patch.set_alpha(0.5)

# reverts
boxplot2 = sns.boxplot(data=revisions_df, x="gender", y="num_reverts", palette=colors, ax=axes[1])
axes[1].set_title("Reverts", fontsize=12)
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("")  
for patch in boxplot2.artists:
    patch.set_alpha(0.5)

# editors
boxplot3 = sns.boxplot(data=revisions_df, x="gender", y="num_editors", palette=colors, ax=axes[2])
axes[2].set_title("Editors", fontsize=12)
axes[2].set_xlabel("")
axes[2].set_ylabel("")  
for patch in boxplot3.artists:
    patch.set_alpha(0.5)

# Adjust grid line transparency
for ax in axes:
    ax.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Leaves space at bottom for the legend
plt.show()

# %%
# FILTER OUT OUTLIERS

# calculate the 99th percentile for each variable
upper_limit_reverts = revisions_df['num_reverts'].quantile(0.99)
upper_limit_revisions = revisions_df['num_revisions'].quantile(0.99)
upper_limit_editors = revisions_df['num_editors'].quantile(0.99)

# filter out values above the 99th percentile
filtered_reverts = revisions_df[revisions_df['num_reverts'] <= upper_limit_reverts]
filtered_revisions = revisions_df[revisions_df['num_revisions'] <= upper_limit_revisions]
filtered_editors = revisions_df[revisions_df['num_editors'] <= upper_limit_editors]

# print updated means and SDs
print("Filtered reverts metrics:")
print(filtered_reverts.groupby("gender")["num_reverts"].agg(['mean', 'std']))

print("\nFiltered revisions metrics:")
print(filtered_revisions.groupby("gender")["num_revisions"].agg(['mean', 'std']))

print("\nFiltered editors metrics:")
print(filtered_editors.groupby("gender")["num_editors"].agg(['mean', 'std']))

# %%
### STATISTICAL TEST FOR REVISIONS, REVERTS, EDITORS

t_stat, p_value = ttest_ind(revisions_df[revisions_df["gender"] == "male"]["num_revisions"], 
                            revisions_df[revisions_df["gender"] == "female"]["num_revisions"], equal_var=False)
print(f"Revisions T-test: t={t_stat:.3f}, p={p_value:.3f}")
# t=-1.872, p=0.062

t_stat, p_value = ttest_ind(revisions_df[revisions_df["gender"] == "male"]["num_reverts"], 
                            revisions_df[revisions_df["gender"] == "female"]["num_reverts"], equal_var=False)
print(f"Reverts T-test: t={t_stat:.3f}, p={p_value:.3f}")
# t=-1.101, p=0.272

t_stat, p_value = ttest_ind(revisions_df[revisions_df["gender"] == "male"]["num_editors"], 
                            revisions_df[revisions_df["gender"] == "female"]["num_editors"], equal_var=False)
print(f"Editors T-test: t={t_stat:.3f}, p={p_value:.3f}")
# t=-2.096, p=0.037

### STATISTICAL TESTS FOR FILTERED DFS

t_stat, p_value = ttest_ind(filtered_revisions[filtered_revisions["gender"] == "male"]["num_revisions"], 
                            filtered_revisions[filtered_revisions["gender"] == "female"]["num_revisions"], equal_var=False)
print(f"Filtered revisions T-test: t={t_stat:.3f}, p={p_value:.3f}")
# t=-2.494, p=0.013

t_stat, p_value = ttest_ind(filtered_reverts[filtered_reverts["gender"] == "male"]["num_reverts"], 
                            filtered_reverts[filtered_reverts["gender"] == "female"]["num_reverts"], equal_var=False)
print(f"Filtered reverts T-test: t={t_stat:.3f}, p={p_value:.3f}")
# t=-2.115, p=0.035

t_stat, p_value = ttest_ind(filtered_editors[filtered_editors["gender"] == "male"]["num_editors"], 
                            filtered_editors[filtered_editors["gender"] == "female"]["num_editors"], equal_var=False)
print(f"Filtered editors T-test: t={t_stat:.3f}, p={p_value:.3f}")
# t=-2.671, p=0.008

# %%

### TEXT ANALYSIS


### LDA TOPIC MODELING

# define stopwords, months, and domain-specific terms
stop_words = set(stopwords.words('english')) | set(string.punctuation)
custom_stopwords = {'new', 'became', 'de', 'first', 'two', 'one', 'also', 'would', 'time', 'year', 'years', 'edit', 
                    'later', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 
                    'august', 'september', 'october', 'november', 'december'}

# merge both stopword sets
stop_words |= custom_stopwords

lemmatizer = WordNetLemmatizer()

def preprocess_for_lda(text):
    if not isinstance(text, str):
        return []

    # convert to lowercase
    text = text.lower()

    # tokenize the text
    tokens = word_tokenize(text)

    # remove possessive 's and quotes
    tokens = [re.sub(r"'s|''", '', token) for token in tokens]

    # remove punctuation attached to words
    tokens = [token.strip(string.punctuation) for token in tokens]

    # remove stopwords, punctuation, numbers, months, domain-specific terms
    tokens = [token for token in tokens if token not in stop_words and not token.isdigit()]

    tokens = [token if re.match(r"[a-zA-Z0-9\s.,;:!?\'\"()-]+", token) else '' for token in tokens]

    # remove any empty tokens (e.g., after punctuation cleaning)
    tokens = [token for token in tokens if token]

    # lemmatize tokens to reduce them to their root form
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


# filter for gender in biographies
female_df = text_df[text_df['gender'] == 'female']
male_df = text_df[text_df['gender'] == 'male']

# apply preprocessing to the 'full_text' column for female and male groups
female_corpus = female_df['plain_content'].apply(preprocess_for_lda)  # This will give a list of tokens
male_corpus = male_df['plain_content'].apply(preprocess_for_lda)  # This will give a list of tokens

# create a dictionary (a mapping from word IDs to words) for each corpus
female_dictionary = corpora.Dictionary(female_corpus)
male_dictionary = corpora.Dictionary(male_corpus)

# create the corpus (Document-Term Matrix) for both female and male data
female_corpus_bow = [female_dictionary.doc2bow(text) for text in female_corpus]
male_corpus_bow = [male_dictionary.doc2bow(text) for text in male_corpus]

# set the number of topics
num_topics = 8

# run LDA 
female_lda_model = LdaModel(corpus=female_corpus_bow, id2word=female_dictionary, num_topics=num_topics, random_state=42)
male_lda_model = LdaModel(corpus=male_corpus_bow, id2word=male_dictionary, num_topics=num_topics, random_state=42)

# %%
# print topics for biographies
print("Female topics:")
for topic in female_lda_model.print_topics(num_words=10):  
    print(topic)

print("\nMale topics:")
for topic in male_lda_model.print_topics(num_words=10):
    print(topic)

# %%
## COUNT WORDS ACROSS TOPICS FOR FREQUENCY ANALYSIS

# extract words from LDA topics
def extract_topic_words(lda_model, num_words=10):
    topic_words = []
    for topic in lda_model.print_topics(num_words=num_words):
        # Extract words from the formatted string
        words = [word.split("*")[1].strip().strip('"') for word in topic[1].split(" + ")]
        topic_words.append(words)
    return topic_words

# get topic words for female and male topics
female_topics = extract_topic_words(female_lda_model, num_words=10)
male_topics = extract_topic_words(male_lda_model, num_words=10)

# print as lists
print("Female topics:", female_topics)
print("Male topics:", male_topics)

# flatten the list of topic words into a single list of words
female_word_list = [word for topic in female_topics for word in topic]
male_word_list = [word for topic in male_topics for word in topic]

# count word frequencies
female_word_counts = Counter(female_word_list)
male_word_counts = Counter(male_word_list)

# print most common words
print("Most common female topic words:", female_word_counts.most_common(15))
print("Most common male topic words:", male_word_counts.most_common(15))

# %%
# combine unique words from both male and female topics
all_words = list(female_word_counts.keys()) + list(male_word_counts.keys())

# create df with the word counts for both female and male topics
word_comparison = pd.DataFrame(
    {
        "Female Frequency": [female_word_counts.get(word, 0) for word in all_words],
        "Male Frequency": [male_word_counts.get(word, 0) for word in all_words],
    },
    index=all_words
)

# remove duplicates (if any) by keeping the first occurrence
word_comparison = word_comparison.loc[~word_comparison.index.duplicated(keep='first')]

# sort by total frequency (sum of female and male frequencies)
word_comparison["Total"] = word_comparison["Female Frequency"] + word_comparison["Male Frequency"]
word_comparison = word_comparison.sort_values(by="Total", ascending=False)

# print top 15 words
print(word_comparison.head(15))


### VISUALIZE FREQUENCY

# select top 15 most frequent words
top_words = word_comparison.head(15).sort_values(by="Female Frequency", ascending=False)

# set style
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 6))
top_words[["Female Frequency", "Male Frequency"]].plot(kind="bar", ax=ax, color=colors, alpha=0.6)

# titles and labels
plt.title("Most frequent words in male vs female topics", fontsize=14)
plt.ylabel("Frequency", fontsize=11)
plt.xticks(rotation=45, fontsize=9)
plt.yticks(fontsize=9)

# legend
plt.legend(["female", "male"], title="Gender", title_fontsize=11, fontsize=9)

# grid
plt.grid(axis="y", linestyle="dashed", alpha=0.3)

# show
plt.show()

# %%
### WORDCLOUDS VISUALIZATION 

# generate word clouds
female_wordcloud = WordCloud(width=800, height=400, colormap="inferno", background_color="white").generate_from_frequencies(female_word_counts)
male_wordcloud = WordCloud(width=800, height=400, colormap="inferno", background_color="white").generate_from_frequencies(male_word_counts)

# plot
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes[0].imshow(female_wordcloud, interpolation="bilinear")
axes[0].set_title("Female Topics", fontsize=16, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(male_wordcloud, interpolation="bilinear")
axes[1].set_title("Male Topics", fontsize=16, fontweight="bold")
axes[1].axis("off")

plt.show()

# %%
### FAMILY RELATED WORDS

# preprocess plain_content as we did before and save it in a new column for easy access

text_df['processed_content'] = text_df['plain_content'].apply(preprocess_for_lda)

# verify result
print(text_df[['plain_content', 'processed_content']].head())
# %%
# and save
text_df.to_csv('final_text.csv', index=False)
# %%
family_related_words = ['mother', 'father', 'daughter', 'son', 
                        'family', 'sister', 'brother', 'parent', 'child', 
                        'husband', 'wife', 'kid', 'grandmother', 
                        'marry', 'divorce', 'widow', 'spouse']

# function to count family-related words in a list of tokens
def count_family_words(tokens, family_words):
    return sum(1 for token in tokens if token in family_words)


# filter text_df by gender
female_df = text_df[text_df['gender'] == 'female'][['title', 'gender', 'processed_content']]
male_df = text_df[text_df['gender'] == 'male'][['title', 'gender', 'processed_content']]

# apply function
female_df['family_word_count'] = female_df['processed_content'].apply(lambda x: count_family_words(x, family_related_words))
male_df['family_word_count'] = male_df['processed_content'].apply(lambda x: count_family_words(x, family_related_words))

# check the average family word count per gender (and SD)
print(female_df['family_word_count'].agg(["mean", "median", "min", "max", "std"]))
print(male_df['family_word_count'].agg(["mean", "median", "min", "max", "std"]))

# %%
# combine the female and male data
family_words_df = pd.concat([
    female_df[['title', 'gender', 'family_word_count']],
    male_df[['title', 'gender', 'family_word_count']]
])
# %%
### STATISTICAL TESTS FOR FAMILY WORD OCCURRENCES

t_stat, p_value = ttest_ind(family_words_df[family_words_df["gender"] == "male"]["family_word_count"], 
                            family_words_df[family_words_df["gender"] == "female"]["family_word_count"], equal_var=False)
print(f"Family word occurrence T-test: t={t_stat:.3f}, p={p_value:.12f}")
# t=-2.494, p=0.013

# %%
### DENSITY PLOT FOR FAMILY WORD COUNTS

plt.figure(figsize=(10, 6))

sns.kdeplot(data=family_words_df, x='family_word_count', hue='gender', fill=True, 
            common_norm=False, alpha=0.4, palette=colors)

plt.xlim(0, family_words_df['family_word_count'].max())

# titles and Labels
plt.title('Density plot of family-related word counts by gender', fontsize=14)
plt.xlabel('Word count', fontsize=11)
plt.ylabel('Density', fontsize=11)

# show
plt.show()

# %%
