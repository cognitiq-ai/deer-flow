# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import re
from typing import Optional
from urllib.parse import urljoin

from markdownify import markdownify as md


class Article:
    url: str

    def __init__(self, title: str, html_content: str, content: Optional[str] = None):
        self.title = title
        self.html_content = html_content
        self._content = content

    @property
    def content(self) -> str:
        if self._content:
            return self._content
        return md(self.html_content)

    def to_markdown(self, including_title: bool = True) -> str:
        markdown = ""
        if including_title:
            markdown += f"# {self.title}\n\n"
        markdown += self.content
        return markdown

    def to_message(self) -> list[dict]:
        image_pattern = r"!\[.*?\]\((.*?)\)"

        content: list[dict[str, str]] = []
        parts = re.split(image_pattern, self.to_markdown())

        for i, part in enumerate(parts):
            if i % 2 == 1:
                image_url = urljoin(self.url, part.strip())
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            else:
                content.append({"type": "text", "text": part.strip()})

        return content
