#!/usr/bin/env python3
"""
PDF Analyzer
------------
Extracts text from PDF files, chunks content, and queries the LLM
for analysis (summaries, insights, Q&A).
Works with Google Gemini API and Groq API via llm_client.py wrapper.
"""

import os
import pdfplumber
from typing import List, Dict, Any
from pathlib import Path

from llm_client import LLMClient


class PDFAnalyzer:
    def __init__(self, provider: str = "gemini", model: str = None):
        """
        Initialize PDF Analyzer with a given provider.

        Args:
            provider (str): "gemini" or "groq"
            model (str): optional model name override
        """
        self.llm = LLMClient(provider=provider, model=model)

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF.

        Args:
            pdf_path (str): path to the PDF file

        Returns:
            str: extracted text
        """
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)

    def chunk_text(self, text: str, max_tokens: int = 1500) -> List[str]:
        """
        Split text into chunks that fit within token limits.

        Args:
            text (str): input text
            max_tokens (int): approximate max tokens per chunk

        Returns:
            List[str]: list of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def analyze(self, pdf_path: str, task: str = "summary") -> Dict[str, Any]:
        """
        Analyze a PDF with the LLM.

        Args:
            pdf_path (str): path to PDF file
            task (str): type of task ("summary", "qa", "insights")

        Returns:
            Dict[str, Any]: analysis results
        """
        full_text = self.extract_text(pdf_path)
        chunks = self.chunk_text(full_text)

        results = []
        for idx, chunk in enumerate(chunks, start=1):
            if task == "summary":
                prompt = (
                    f"Summarize the following part of a document (part {idx}):\n\n{chunk}"
                )
            elif task == "insights":
                prompt = (
                    f"Extract key insights, patterns, and anomalies from this part of a document (part {idx}):\n\n{chunk}"
                )
            elif task == "qa":
                prompt = (
                    f"Answer questions based only on the following document section (part {idx}):\n\n{chunk}"
                )
            else:
                prompt = f"Analyze the following text:\n\n{chunk}"

            response = self.llm.generate(prompt)
            results.append({"chunk": idx, "response": response})

        return {"task": task, "results": results}

    def qa(self, pdf_path: str, question: str) -> str:
        """
        Ask a question about a PDF.

        Args:
            pdf_path (str): path to PDF file
            question (str): question to ask

        Returns:
            str: answer from the LLM
        """
        full_text = self.extract_text(pdf_path)
        chunks = self.chunk_text(full_text)

        answers = []
        for idx, chunk in enumerate(chunks, start=1):
            prompt = (
                f"Answer the following question based on the text (part {idx}):\n"
                f"Question: {question}\n\n"
                f"Text:\n{chunk}"
            )
            response = self.llm.generate(prompt)
            answers.append(response)

        final_prompt = (
            f"Here are multiple answers extracted from different chunks of a PDF:\n\n"
            + "\n\n".join(answers)
            + f"\n\nProvide a consolidated and accurate answer to the question: {question}"
        )
        return self.llm.generate(final_prompt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze PDFs with LLM")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument(
        "--task", choices=["summary", "insights", "qa"], default="summary", help="Task type"
    )
    parser.add_argument("--provider", choices=["gemini", "groq"], default="gemini")
    parser.add_argument("--question", help="Question to ask (only for qa task)")

    args = parser.parse_args()

    analyzer = PDFAnalyzer(provider=args.provider)
    if args.task == "qa":
        if not args.question:
            raise ValueError("Question required for QA task")
        answer = analyzer.qa(args.pdf_path, args.question)
        print("\n=== Final Answer ===\n")
        print(answer)
    else:
        results = analyzer.analyze(args.pdf_path, task=args.task)
        for r in results["results"]:
            print(f"\n--- Chunk {r['chunk']} ---\n")
            print(r["response"])
