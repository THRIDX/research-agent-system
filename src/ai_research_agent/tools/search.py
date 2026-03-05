"""Search tools: arxiv, GitHub, and web search."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import arxiv  # type: ignore[import-untyped]
import requests


@dataclass
class ArxivResult:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    pdf_url: str
    categories: list[str] = field(default_factory=list)


def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
) -> list[ArxivResult]:
    """Search arxiv for papers matching query."""
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)
    results: list[ArxivResult] = []
    for paper in client.results(search):
        results.append(
            ArxivResult(
                arxiv_id=paper.entry_id.split("/")[-1],
                title=paper.title,
                authors=[a.name for a in paper.authors],
                abstract=paper.summary,
                published=paper.published.isoformat() if paper.published else "",
                pdf_url=paper.pdf_url or "",
                categories=paper.categories,
            )
        )
    return results


@dataclass
class GithubRepo:
    full_name: str
    description: str
    url: str
    stars: int
    language: Optional[str]
    topics: list[str] = field(default_factory=list)


def search_github(
    query: str,
    max_results: int = 10,
    github_token: Optional[str] = None,
) -> list[GithubRepo]:
    """Search GitHub repositories matching query."""
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    resp = requests.get(
        "https://api.github.com/search/repositories",
        params={"q": query, "per_page": min(max_results, 100), "sort": "stars"},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])

    results: list[GithubRepo] = []
    for item in items[:max_results]:
        results.append(
            GithubRepo(
                full_name=item["full_name"],
                description=item.get("description") or "",
                url=item["html_url"],
                stars=item.get("stargazers_count", 0),
                language=item.get("language"),
                topics=item.get("topics", []),
            )
        )
    return results


@dataclass
class SemanticScholarPaper:
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    year: Optional[int]
    citation_count: int
    url: str


def search_semantic_scholar(
    query: str,
    max_results: int = 10,
    fields: Optional[list[str]] = None,
) -> list[SemanticScholarPaper]:
    """Search Semantic Scholar for papers."""
    if fields is None:
        fields = ["paperId", "title", "authors", "abstract", "year", "citationCount", "url"]

    resp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": min(max_results, 100), "fields": ",".join(fields)},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])

    results: list[SemanticScholarPaper] = []
    for item in data:
        results.append(
            SemanticScholarPaper(
                paper_id=item.get("paperId", ""),
                title=item.get("title", ""),
                authors=[a.get("name", "") for a in item.get("authors", [])],
                abstract=item.get("abstract") or "",
                year=item.get("year"),
                citation_count=item.get("citationCount", 0),
                url=item.get("url", ""),
            )
        )
        time.sleep(0.1)  # be polite to the API
    return results
