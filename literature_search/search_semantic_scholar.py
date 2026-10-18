"""Updated literature search for the revision (Semantic Scholar bulk search API).

The original search (Google Scholar, accessed 25 August 2024) was extended to
work published from 2024 onwards.  Each query is a boolean phrase query over
titles and abstracts (Semantic Scholar syntax: "phrase", + for AND, | for OR).
The script writes

  search_log.csv        query id, query, year filter, access date, number of hits
  records_<id>.csv      the 100 most cited records of each query (screened by
                        title and abstract)

    python search_semantic_scholar.py
"""

import csv
import datetime
import json
import time
import urllib.parse
import urllib.request

QUERIES = [
    ("Q1", '"catastrophic forgetting" + "continual learning"'),
    ("Q2", '"continual learning" + ("large language model" | "large language models")'),
    ("Q3", '"continual learning" + ("vision-language" | "foundation model" | "pre-trained model")'),
    ("Q4", '"continual learning" + ("spiking neural network" | "spiking neural networks" | neuromorphic)'),
    ("Q5", '(metaplasticity | neuromodulation | "homeostatic plasticity" | neurogenesis) + forgetting'),
    ("Q6", '"class-incremental learning"'),
    ("Q7", '("knowledge editing" | "model editing") + (lifelong | sequential | continual)'),
    ("Q8", '"continual learning" + (hippocampus | sleep | "complementary learning systems")'),
]
YEARS = "2024-2026"
API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"


def get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cl-review-search"}), timeout=60))
        except Exception:
            time.sleep(10 * (attempt + 1))  # the shared, unauthenticated rate limit is low
    raise RuntimeError("Semantic Scholar request failed: " + url)


def main():
    today = datetime.date.today().isoformat()
    with open("search_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "query", "years", "accessed", "hits"])
        for qid, q in QUERIES:
            d = get({"query": q, "year": YEARS, "fields": "title,year,venue,citationCount,externalIds",
                     "sort": "citationCount:desc"})
            w.writerow([qid, q, YEARS, today, d.get("total", 0)])
            with open(f"records_{qid}.csv", "w", newline="", encoding="utf-8") as g:
                r = csv.writer(g)
                r.writerow(["rank", "year", "title", "venue", "citations", "doi", "arxiv"])
                for k, p in enumerate((d.get("data") or [])[:100], 1):
                    ids = p.get("externalIds") or {}
                    r.writerow([k, p.get("year"), p.get("title"), p.get("venue"), p.get("citationCount"), ids.get("DOI", ""), ids.get("ArXiv", "")])
            print(qid, d.get("total"), flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
