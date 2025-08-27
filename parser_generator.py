#!/usr/bin/env python3
"""
Parser Generator
Uses LLM to create custom Python parser classes for each bank's PDF format.
"""

import os
from pathlib import Path
from llm_client import LLMClient

CUSTOM_PARSERS_DIR = Path("custom_parsers")


class ParserGenerator:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        CUSTOM_PARSERS_DIR.mkdir(exist_ok=True)

    def _parser_filename(self, bank: str) -> Path:
        return CUSTOM_PARSERS_DIR / f"{bank}_parser.py"

    def generate_parser(self, bank: str, pdf_structure: str) -> Path:
        """
        Generate a parser for the given bank if not already present.
        """
        parser_path = self._parser_filename(bank)

        if parser_path.exists():
            print(f"[INFO] Parser already exists for {bank}: {parser_path}")
            return parser_path

        prompt = f"""
You are an expert in financial data extraction.

Generate a Python parser class for the bank "{bank}".
The parser must:
- Be named {bank.capitalize()}Parser.
- Inherit from BaseParser (assume already available).
- Implement parse(self, pdf_path: str) -> pandas.DataFrame
- Use camelCase method names only where necessary.
- Input: PDF file path
- Output: DataFrame with columns: date, description, debit, credit, balance.
- Extract transaction tables from the PDF text or structure.

Here is the extracted PDF structure (simplified):
{pdf_structure}
"""

        code = self.llm.generate(prompt)

        with open(parser_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated parser\n")
            f.write("import pandas as pd\n")
            f.write("from base_parser import BaseParser\n\n")
            f.write(code.strip() + "\n")

        print(f"[INFO] Generated parser for {bank}: {parser_path}")
        return parser_path
