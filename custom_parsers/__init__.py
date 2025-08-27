"""
Custom parsers package for bank statement PDFs
Generated parsers are saved here by the AI Agent
"""

from pathlib import Path
import importlib.util
from typing import Optional, Callable
import pandas as pd

def load_parser(bank_name: str) -> Optional[Callable[[str], pd.DataFrame]]:
    """
    Dynamically load a parser for a specific bank
    
    Args:
        bank_name: Name of the bank (e.g., 'icici', 'sbi')
        
    Returns:
        Parse function or None if not found
    """
    parser_file = Path(__file__).parent / f"{bank_name}_parser.py"
    
    if not parser_file.exists():
        return None
    
    try:
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(f"{bank_name}_parser", parser_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Return the parse function
        return getattr(module, 'parse', None)
        
    except Exception as e:
        print(f"Error loading parser for {bank_name}: {e}")
        return None

def list_available_parsers() -> list:
    """List all available parsers in this directory"""
    parsers_dir = Path(__file__).parent
    parser_files = list(parsers_dir.glob("*_parser.py"))
    return [f.stem.replace("_parser", "") for f in parser_files]

__all__ = ['load_parser', 'list_available_parsers']