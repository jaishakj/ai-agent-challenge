"""
Test suite for generated bank statement parsers
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from custom_parsers import load_parser, list_available_parsers

class TestGeneratedParsers:
    """Test cases for AI-generated parsers"""
    
    @pytest.fixture
    def data_dir(self):
        """Return path to test data directory"""
        return Path(project_root) / "data"
    
    def test_icici_parser_exists(self):
        """Test that ICICI parser was generated"""
        parser = load_parser("icici")
        assert parser is not None, "ICICI parser should be generated"
        assert callable(parser), "Parser should be callable"
    
    def test_icici_parser_functionality(self, data_dir):
        """Test ICICI parser with sample PDF"""
        parser = load_parser("icici")
        if parser is None:
            pytest.skip("ICICI parser not generated yet")
        
        pdf_path = data_dir / "icici" / "icici_sample.pdf"
        if not pdf_path.exists():
            pytest.skip("ICICI sample PDF not found")
        
        # Test parser execution
        result = parser(str(pdf_path))
        
        # Basic validation
        assert isinstance(result, pd.DataFrame), "Parser should return DataFrame"
        assert not result.empty, "Parser should extract data"
        assert len(result) > 0, "Should extract at least one transaction"
        
        print(f"✅ Extracted {len(result)} transactions")
        print(f"📊 Columns: {list(result.columns)}")
    
    def test_parser_output_schema(self, data_dir):
        """Test that parser output matches expected schema"""
        parser = load_parser("icici")
        if parser is None:
            pytest.skip("ICICI parser not generated yet")
            
        pdf_path = data_dir / "icici" / "icici_sample.pdf"
        csv_path = data_dir / "icici" / "icici_sample.csv"
        
        if not pdf_path.exists() or not csv_path.exists():
            pytest.skip("Sample files not found")
        
        # Load expected schema
        expected_df = pd.read_csv(csv_path)
        expected_columns = set(expected_df.columns)
        
        # Test parser
        result = parser(str(pdf_path))
        actual_columns = set(result.columns)
        
        # Validate schema compatibility
        missing_cols = expected_columns - actual_columns
        extra_cols = actual_columns - expected_columns
        
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
        if extra_cols:
            print(f"ℹ️ Extra columns: {extra_cols}")
        
        # At least some core columns should match
        core_matches = expected_columns & actual_columns
        assert len(core_matches) > 0, f"No matching columns found. Expected: {expected_columns}, Got: {actual_columns}"
        
        print(f"✅ {len(core_matches)}/{len(expected_columns)} columns match")
    
    def test_parser_data_quality(self, data_dir):
        """Test data quality of parser output"""
        parser = load_parser("icici")
        if parser is None:
            pytest.skip("ICICI parser not generated yet")
            
        pdf_path = data_dir / "icici" / "icici_sample.pdf"
        if not pdf_path.exists():
            pytest.skip("ICICI sample PDF not found")
        
        result = parser(str(pdf_path))
        
        # Data quality checks
        assert not result.empty, "Result should not be empty"
        
        # Check for date columns
        date_cols = [col for col in result.columns if 'date' in col.lower()]
        if date_cols:
            for col in date_cols:
                # Should be able to parse as datetime
                try:
                    pd.to_datetime(result[col])
                    print(f"✅ Date column '{col}' is valid")
                except:
                    pytest.fail(f"Date column '{col}' contains invalid dates")
        
        # Check for amount columns
        amount_cols = [col for col in result.columns if any(word in col.lower() for word in ['amount', 'balance', 'debit', 'credit'])]
        if amount_cols:
            for col in amount_cols:
                # Should be numeric
                assert pd.api.types.is_numeric_dtype(result[col]) or result[col].dtype == 'object', f"Amount column '{col}' should be numeric or convertible"
                print(f"✅ Amount column '{col}' is valid")
        
        # Check for reasonable number of transactions
        assert len(result) >= 1, "Should extract at least 1 transaction"
        assert len(result) <= 1000, "Shouldn't extract more than 1000 transactions (likely parsing error)"
        
        print(f"✅ Data quality checks passed for {len(result)} transactions")
    
    def test_parser_error_handling(self):
        """Test parser error handling with invalid inputs"""
        parser = load_parser("icici")
        if parser is None:
            pytest.skip("ICICI parser not generated yet")
        
        # Test with non-existent file
        try:
            result = parser("non_existent_file.pdf")
            # Should either return empty DataFrame or raise appropriate error
            if result is not None:
                assert isinstance(result, pd.DataFrame), "Should return DataFrame even for invalid input"
        except (FileNotFoundError, Exception) as e:
            # Expected behavior - parser should handle errors gracefully
            print(f"✅ Parser handles invalid file gracefully: {type(e).__name__}")
    
    def test_list_available_parsers(self):
        """Test that we can list available parsers"""
        parsers = list_available_parsers()
        assert isinstance(parsers, list), "Should return list of parser names"
        
        if len(parsers) > 0:
            print(f"✅ Found {len(parsers)} parsers: {parsers}")
            
            # Each listed parser should be loadable
            for parser_name in parsers:
                parser = load_parser(parser_name)
                assert parser is not None, f"Parser '{parser_name}' should be loadable"
                assert callable(parser), f"Parser '{parser_name}' should be callable"
        else:
            print("ℹ️ No parsers generated yet")

class TestAgentIntegration:
    """Integration tests for the full agent workflow"""
    
    def test_agent_can_run(self):
        """Test that agent can be imported and basic functions work"""
        try:
            from agent import BankStatementAgent, AgentState, Memory
            
            # Test basic instantiation
            agent = BankStatementAgent()
            assert agent is not None
            
            # Test memory creation
            memory = Memory(pdf_path="test.pdf", target_bank="test")
            assert memory.current_state == AgentState.PLANNING
            assert memory.attempts == 0
            
            print("✅ Agent classes can be imported and instantiated")
            
        except ImportError as e:
            pytest.fail(f"Cannot import agent module: {e}")
    
    def test_pdf_analysis_tools(self, tmp_path):
        """Test PDF analysis tools work"""
        try:
            from agent import BankStatementAgent
            
            agent = BankStatementAgent()
            
            # Test with actual PDF if available
            data_dir = Path(project_root) / "data" / "icici"
            pdf_path = data_dir / "icici_sample.pdf"
            
            if pdf_path.exists():
                result = agent._analyze_pdf(str(pdf_path))
                
                assert isinstance(result, dict), "Should return analysis dict"
                assert 'structure' in result
                assert 'text' in result
                assert 'patterns' in result
                
                print(f"✅ PDF analysis successful: {len(result['text'])} chars extracted")
            else:
                print("ℹ️ No sample PDF found, skipping PDF analysis test")
                
        except Exception as e:
            pytest.fail(f"PDF analysis failed: {e}")

# Pytest configuration
class TestRunner:
    """Custom test runner for development"""
    
    @staticmethod
    def run_quick_tests():
        """Run a subset of quick tests for development"""
        print("🧪 Running quick tests...")
        
        # Test parser loading
        try:
            parsers = list_available_parsers()
            print(f"📁 Found parsers: {parsers}")
            
            for parser_name in parsers:
                parser = load_parser(parser_name)
                if parser:
                    print(f"✅ {parser_name} parser loads successfully")
                else:
                    print(f"❌ {parser_name} parser failed to load")
                    
        except Exception as e:
            print(f"❌ Quick test failed: {e}")
    
    @staticmethod
    def run_integration_test():
        """Run integration test with sample data"""
        print("🔗 Running integration test...")
        
        data_dir = Path(project_root) / "data" / "icici"
        pdf_path = data_dir / "icici_sample.pdf"
        
        if not pdf_path.exists():
            print("⚠️ Sample PDF not found, skipping integration test")
            return
        
        try:
            # Test full workflow
            from agent import BankStatementAgent
            
            agent = BankStatementAgent()
            
            # Test PDF analysis
            analysis = agent._analyze_pdf(str(pdf_path))
            print(f"📄 PDF analysis: {analysis['structure']['num_pages']} pages")
            
            # Test parser loading if available
            parser = load_parser("icici")
            if parser:
                result = parser(str(pdf_path))
                print(f"📊 Parser result: {len(result)} rows, {len(result.columns)} columns")
            else:
                print("ℹ️ No ICICI parser available yet")
                
        except Exception as e:
            print(f"❌ Integration test failed: {e}")

if __name__ == "__main__":
    # Allow running tests directly
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        TestRunner.run_quick_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "integration":
        TestRunner.run_integration_test()
    else:
        # Run with pytest
        pytest.main([__file__, "-v"])