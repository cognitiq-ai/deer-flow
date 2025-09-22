import json
from typing import Any, Dict, List, Optional

from langchain_community.tools.searx_search.tool import SearxSearchResults
from langchain_community.utilities import SearxSearchWrapper


class CustomSearxSearchWrapper(SearxSearchWrapper):
    """Custom SearxNG search wrapper, return standardized JSON format results"""

    def results(
        self,
        query: str,
        num_results: int,
        engines: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        query_suffix: Optional[str] = "",
        **kwargs: Any,
    ) -> List[Dict]:
        # Call parent class to get raw results
        raw_results = super().results(
            query=query,
            num_results=num_results,
            engines=engines,
            categories=categories,
            query_suffix=query_suffix,
            **kwargs,
        )
        summary = super().run(query)

        # Convert asynchronous results format
        formatted_results = []
        for result in raw_results:
            formatted = {
                "type": "page",
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "content": result.get("snippet", ""),
            }
            formatted_results.append(formatted)

        return formatted_results

    async def aresults(
        self,
        query: str,
        num_results: int,
        engines: Optional[List[str]] = None,
        query_suffix: Optional[str] = "",
        **kwargs: Any,
    ) -> str:
        """Asynchronously get and format results"""
        raw_results = await super().aresults(
            query=query,
            num_results=num_results,
            engines=engines,
            query_suffix=query_suffix,
            **kwargs,
        )
        summary = await super().arun(query)

        # Convert asynchronous results format
        formatted_results = []
        for i, result in enumerate(raw_results):
            formatted = {
                "type": "page",
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "content": result.get("snippet", ""),
            }
            formatted_results.append(formatted)

        return formatted_results


class CustomSearxSearchResults(SearxSearchResults):
    """SearxNG search tool that supports JSON format output"""

    def _run(self, query: str) -> str:
        """Synchronously execute search and return JSON string"""
        results = self.wrapper.results(
            query=query, num_results=self.kwargs.get("num_results", 5)
        )
        return results

    async def _arun(self, query: str) -> str:
        """Asynchronously execute search and return JSON string"""
        results = await self.wrapper.aresults(
            query=query, num_results=self.kwargs.get("num_results", 5)
        )
        return results
