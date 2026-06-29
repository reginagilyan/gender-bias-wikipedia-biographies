# Gender Bias in Wikipedia Biographies (Vital Articles)

This project, developed as part of a computational social science coursework, investigates whether gender disparities documented in Wikipedia persist even within its most curated and quality-controlled article set. Focusing on the English Wikipedia Vital Articles (Level 4 – People) list (~1,997 biographies), it applies large-scale web scraping, API integration, and natural language processing techniques to construct a structured dataset of biographies. By combining Wikipedia and Wikidata APIs with asynchronous data pipelines, the project enables scalable extraction of article content, metadata, and revision histories, supporting statistical and text-based comparisons between male and female biographies within this highly curated subset.

## Data & Methods

### Data collection
- Extracted Wikipedia Vital Articles list (titles, links, quality ratings) via HTTP requests and HTML parsing using BeautifulSoup
- Retrieved full article text, section structure, and internal hyperlinks via the MediaWiki API (action=parse)
- Collected full revision histories and editor metadata via the Wikipedia API (revisions endpoint)
- Queried Wikidata API for gender classification using the P21 property

### Processing
- HTML cleaning and parsing using BeautifulSoup (removal of tables, references, and non-content elements)
- NLP preprocessing with NLTK (tokenization, stopword removal, lemmatization)
- Construction of structured datasets across article, revision, and text levels using pandas
- Asynchronous data pipeline using asyncio and aiohttp for parallel API requests, including rate-limit handling and retries

### Analysis
- Structural comparison of biographies across gender groups using measures of article length (word count), structural complexity (number of sections), and connectivity (internal hyperlinks) to assess differences in how extensively articles are developed
- Editorial activity analysis based on revision histories, including total edits, reverts (as a proxy for content disputes), number of unique editors, and revert rates to capture patterns of engagement and editorial scrutiny
- Article quality analysis using Wikipedia’s internal rating system (Start-Class to Featured Articles) to examine whether visibility and quality distribution differ by gender
- Topic modeling (LDA, gensim) applied separately to male and female biographies to identify dominant themes in the textual content and compare how biographies are framed across gender groups
- Lexical analysis of family- and relationship-related terminology to quantify differences in personal and domestic framing of biographies
- Statistical testing (Welch’s t-tests) used to assess whether observed differences between gender groups are statistically significant across structural, editorial, and textual features


### Key note
This is an **exploratory computational text analysis** of Wikipedia biographies. Gender is derived from Wikidata (P21 property) and should be interpreted as a structured metadata label rather than a sociological ground truth.
