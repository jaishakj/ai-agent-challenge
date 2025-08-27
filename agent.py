#!/usr/bin/env python3
"""
AI Agent for Bank Statement PDF Parsing
Updated: uses modular pdf_analyzer + llm_client, 
fixes for LLM model handling, response parsing, error visibility, and CLI model override.
"""

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pdfplumber
import re
from dataclasses import dataclass
from enum import Enum

# Internal imports
from pdf_analyzer import PDFAnalyzer
from llm_client import LLMClient

from dotenv import load_dotenv

load_dotenv()


class AgentState(Enum):
    PLANNING = "planning"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    TESTING = "testing"
    REFINING = "refining"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Memory:
    pdf_path: str
    target_bank: str
    attempts: int = 0
    max_attempts: int = 3
    current_state: AgentState = AgentState.PLANNING
    pdf_structure: Optional[Dict] = None
    extracted_text: Optional[str] = None
    patterns: Optional[Dict] = None
    sample_csv_data: Optional[pd.DataFrame] = None
    current_parser: Optional[str] = None
    parser_history: List[str] = None
    last_result: Optional[pd.DataFrame] = None
    validation_errors: List[str] = None

    def __post_init__(self):
        if self.parser_history is None:
            self.parser_history = []
        if self.validation_errors is None:
            self.validation_errors = []


class BankStatementAgent:
    def __init__(self, llm_provider: str = "groq", model_override: Optional[str] = None):
        self.memory: Optional[Memory] = None
        self.llm_provider = llm_provider

        # Initialize LLM client wrapper
        self.client = LLMClient(provider=llm_provider, model=model_override)

        self.tools = {
            "pdf_analyzer": self._analyze_pdf,
            "schema_loader": self._load_expected_schema,
            "code_generator": self._generate_parser,
            "test_runner": self._test_parser,
            "validator": self._validate_output,
        }

    def run(self, pdf_path: str, target_bank: str) -> Optional[pd.DataFrame]:
        self.memory = Memory(pdf_path=pdf_path, target_bank=target_bank)

        print(f"🤖 Agent starting: parsing {target_bank} statement from {pdf_path}")

        try:
            self.memory.sample_csv_data = self.tools["schema_loader"](target_bank)
            if self.memory.sample_csv_data is not None:
                print(f"📋 Loaded expected schema with columns: {list(self.memory.sample_csv_data.columns)}")
        except Exception as e:
            print(f"⚠️ Could not load expected schema: {e}")
            traceback.print_exc()

        while (
            self.memory.current_state not in [AgentState.COMPLETE, AgentState.FAILED]
            and self.memory.attempts < self.memory.max_attempts
        ):
            self.memory.attempts += 1
            print(f"\n📍 Attempt {self.memory.attempts}/{self.memory.max_attempts}")
            print(f"🧠 State: {self.memory.current_state.value}")

            try:
                if self.memory.current_state == AgentState.PLANNING:
                    self._plan()
                elif self.memory.current_state == AgentState.ANALYZING:
                    self._analyze()
                elif self.memory.current_state == AgentState.GENERATING:
                    self._generate()
                elif self.memory.current_state == AgentState.TESTING:
                    self._test()
                elif self.memory.current_state == AgentState.REFINING:
                    self._refine()
            except Exception as e:
                print(f"❌ Error in {self.memory.current_state.value}: {e}")
                traceback.print_exc()
                self.memory.validation_errors.append(f"State {self.memory.current_state.value}: {str(e)}")
                self.memory.current_state = AgentState.REFINING

        if self.memory.current_state == AgentState.COMPLETE:
            print("✅ Agent completed successfully!")
            self._save_parser()
            return self.memory.last_result
        else:
            print("❌ Agent failed to generate working parser")
            for e in (self.memory.validation_errors or [])[-10:]:
                print("•", e)
            return None

    def _plan(self):
        print("📋 Planning parsing strategy...")
        if self.memory.sample_csv_data is not None:
            print(f"📊 Target columns: {list(self.memory.sample_csv_data.columns)}")
        self.memory.current_state = AgentState.ANALYZING

    def _analyze(self):
        print("🔍 Analyzing PDF structure...")
        analyzer = PDFAnalyzer(self.memory.pdf_path)
        analysis_text = analyzer.extract_structure()
        self.memory.pdf_structure = {"summary": "auto-extracted", "details": analysis_text[:1000]}
        self.memory.extracted_text = analysis_text
        self.memory.patterns = {"line_count": len(analysis_text.splitlines())}
        self.memory.current_state = AgentState.GENERATING

    def _generate(self):
        print("⚡ Generating parser code...")
        parser_code = self.tools["code_generator"](
            self.memory.pdf_structure,
            self.memory.patterns,
            self.memory.target_bank,
            self.memory.sample_csv_data,
            (self.memory.extracted_text or "")[:2000],
        )
        self.memory.current_parser = parser_code
        self.memory.parser_history.append(parser_code)
        self.memory.current_state = AgentState.TESTING

    def _test(self):
        print("🧪 Testing parser...")
        try:
            result = self.tools["test_runner"](self.memory.current_parser, self.memory.pdf_path)
            self.memory.last_result = result
            if result is not None and not result.empty:
                validation_errors = self.tools["validator"](result, self.memory.sample_csv_data)
                self.memory.validation_errors.extend(validation_errors)
                if not validation_errors:
                    self.memory.current_state = AgentState.COMPLETE
                else:
                    self.memory.current_state = AgentState.REFINING
            else:
                msg = "Empty result from parser"
                print("⚠️", msg)
                self.memory.validation_errors.append(msg)
                self.memory.current_state = AgentState.REFINING
        except Exception as e:
            print("❌ Exception during testing:", e)
            tb = traceback.format_exc()
            print(tb)
            self.memory.validation_errors.append(f"Test error: {str(e)}\n{tb}")
            self.memory.current_state = AgentState.REFINING

    def _refine(self):
        print("🔧 Refining approach...")
        if self.memory.attempts >= self.memory.max_attempts:
            self.memory.current_state = AgentState.FAILED
        else:
            self.memory.current_state = AgentState.GENERATING

    def _save_parser(self):
        if not self.memory.current_parser:
            return
        output_dir = Path("custom_parsers")
        output_dir.mkdir(exist_ok=True)
        parser_file = output_dir / f"{self.memory.target_bank}_parser.py"
        header = f'''"""
Generated parser for {self.memory.target_bank.upper()} bank statements
Auto-generated by AI Agent
Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
'''
        with open(parser_file, "w", encoding="utf-8") as f:
            f.write(header + self.memory.current_parser)
        print(f"💾 Saved parser to {parser_file}")

    def _load_expected_schema(self, bank: str) -> Optional[pd.DataFrame]:
        csv_path = Path(f"data/{bank}/{bank}_sample.csv")
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                print("⚠️ Failed to read sample CSV:", e)
                traceback.print_exc()
        return None

    def _generate_parser(
        self,
        structure: Dict,
        patterns: Dict,
        bank: str,
        sample_csv: Optional[pd.DataFrame],
        sample_text: str,
    ) -> str:
        target_columns = (
            list(sample_csv.columns) if sample_csv is not None else ["date", "description", "amount", "balance"]
        )
        prompt = f"""
You are an expert Python developer.
Write ONLY valid Python code that defines a function:

def parse(pdf_path: str) -> pd.DataFrame

Requirements:
- Use pdfplumber to open and iterate pages.
- Extract transaction-like lines and return a pandas DataFrame.
- Ensure DataFrame has columns: {target_columns}
- Use robust parsing: coerce dates with pd.to_datetime(..., errors='coerce'), clean numeric strings (remove commas), and handle missing values.
- Do NOT include any prose, markdown, or explanation. Return only the Python function code.

Context:
- PDF text sample (first 2000 chars): {sample_text}
- PDF structure summary: {structure}
"""
        try:
            code = self.client.generate(prompt)
            if not code:
                raise RuntimeError("LLM returned no code content")

            if code.startswith("```"):
                code = re.sub(r"^```(?:python)?\s*", "", code)
                code = re.sub(r"\s*```$", "", code)

            if "def parse(" not in code:
                raise RuntimeError("Generated code does not define parse()")

            return code
        except Exception as e:
            err_msg = f"⚠️ LLM generation failed: {e}"
            print(err_msg)
            traceback.print_exc()
            if self.memory:
                self.memory.validation_errors.append(err_msg + "\n" + traceback.format_exc())
            return self._get_template_parser(bank, sample_csv)

    def _get_template_parser(self, bank: str, sample_csv: Optional[pd.DataFrame]) -> str:
        columns = (
            ["date", "description", "amount", "balance"]
            if sample_csv is None
            else list(sample_csv.columns)
        )
        return f'''
def parse(pdf_path: str) -> pd.DataFrame:
    import pandas as pd, pdfplumber, re
    transactions = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.split("\\n"):
                if re.search(r"\\d{{1,2}}[/-]\\d{{1,2}}[/-]\\d{{2,4}}", line):
                    parts = line.split()
                    if len(parts) >= 2:
                        date_str = parts[0]
                        rest = " ".join(parts[1:])
                        m = re.findall(r"[-+]?[0-9]+(?:[,0-9]*\\.[0-9]+)?", rest)
                        amt = None
                        if m:
                            amt = m[-1].replace(",", "")
                        try:
                            transactions.append({{
                                "{columns[0]}": pd.to_datetime(date_str, errors="coerce"),
                                "{columns[1]}": rest,
                                "{columns[2]}": float(amt) if amt else None,
                            }})
                        except Exception:
                            continue
    df = pd.DataFrame(transactions)
    for c in {columns}:
        if c not in df.columns:
            df[c] = None
    return df
'''

    def _test_parser(self, parser_code: str, pdf_path: str) -> Optional[pd.DataFrame]:
        namespace = {"pd": pd, "pdfplumber": pdfplumber, "re": re, "__builtins__": __builtins__}
        try:
            exec(parser_code, namespace)
            parse_func = namespace.get("parse")
            if not parse_func:
                raise RuntimeError("parse() function not found after exec")
            return parse_func(pdf_path)
        except Exception as e:
            err = f"Parser execution failed: {e}"
            print("❌", err)
            traceback.print_exc()
            if self.memory:
                self.memory.validation_errors.append(err + "\n" + traceback.format_exc())
            return None

    def _validate_output(self, df: pd.DataFrame, sample_csv: Optional[pd.DataFrame]) -> List[str]:
        errors = []
        if df is None or df.empty:
            errors.append("Parser returned empty DataFrame")
            return errors
        if sample_csv is not None:
            expected = list(sample_csv.columns)
            missing = set(expected) - set(df.columns)
            if missing:
                errors.append(f"Missing columns: {missing}")
        return errors


def main():
    parser = argparse.ArgumentParser(description="AI Agent Bank Statement Parser")
    parser.add_argument("--target", required=True, help="Target bank (e.g., icici)")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--llm", default="groq", choices=["groq", "google"], help="LLM provider")
    parser.add_argument("--model", default=None, help="Optional model identifier to override defaults")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"❌ PDF not found: {args.pdf_path}")
        sys.exit(1)
    if args.llm == "groq" and not os.getenv("GROQ_API_KEY"):
        print("❌ Missing GROQ_API_KEY in .env")
        sys.exit(1)
    if args.llm == "google" and not os.getenv("GOOGLE_API_KEY"):
        print("❌ Missing GOOGLE_API_KEY in .env")
        sys.exit(1)

    agent = BankStatementAgent(llm_provider=args.llm, model_override=args.model)
    result = agent.run(args.pdf_path, args.target)
    if result is not None and not result.empty:
        print(f"\n📊 Extracted {len(result)} rows")
        out_file = f"output_{args.target}_result.csv"
        result.to_csv(out_file, index=False)
        print(f"💾 Saved results to {out_file}") 
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
