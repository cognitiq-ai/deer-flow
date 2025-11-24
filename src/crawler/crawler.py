# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT


from cleantext import clean
from trafilatura.downloads import fetch_response

from src.config import SELECTED_CRAWL_ENGINE, CrawlEngine
from tika import parser

from .article import Article
from .jina_client import JinaClient
from .readability_extractor import ReadabilityExtractor


class Crawler:
    def crawl(self, url: str) -> Article:
        if SELECTED_CRAWL_ENGINE == CrawlEngine.TIKA.value:
            article = self.tika_extractor(url)
        elif SELECTED_CRAWL_ENGINE == CrawlEngine.JINA.value:
            article = self.jina_extractor(url)
        else:
            raise ValueError(f"Invalid crawl engine: {SELECTED_CRAWL_ENGINE}")

        article.url = url
        return article

    def tika_extractor(self, url: str) -> Article:
        """
        Extract content from a URL using apache tika.
        """
        # Get the raw response
        response = fetch_response(url)
        # Parse the buffer content
        if response is None:
            return Article(title="", content="")
        parsed = parser.from_buffer(response.data)
        content = clean(parsed["content"], lower=False)[:50000]
        title = parsed["metadata"].get("dc.title", "")
        return Article(title=title, content=content)

    def jina_extractor(self, url: str) -> Article:
        """
        Extract content from a URL using jina.
        """
        # To help LLMs better understand content, we extract clean
        # articles from HTML, convert them to markdown, and split
        # them into text and image blocks for one single and unified
        # LLM message.
        #
        # Jina is not the best crawler on readability, however it's
        # much easier and free to use.
        #
        # Instead of using Jina's own markdown converter, we'll use
        # our own solution to get better readability results.
        jina_client = JinaClient()
        html = jina_client.crawl(url, return_format="html")
        extractor = ReadabilityExtractor()
        article = extractor.extract_article(html)
        return article
