"""
LLM Tool Interface for Mathematical Analysis

This module provides a clean interface for LLMs to use mathematical tools
directly, with examples and helper functions.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
import logging
import re

# Import the TypeScriptTools directly
from python_agent.src.tools.test_tools import TypeScriptTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLMToolInterface")

class MathToolInterface:
    """
    Interface for LLMs to use mathematical tools directly.
    
    This class provides tools for mathematical and statistical analysis,
    with methods designed to be easily used by LLMs.
    """
    
    @staticmethod
    def get_tool_documentation() -> str:
        """
        Get documentation for the available tools in a format suitable for LLMs.
        
        Returns:
            str: Tool documentation
        """
        docs = """
AVAILABLE MATHEMATICAL TOOLS

1. mean(numbers: List[float]) -> float
   - Description: Calculate the arithmetic mean (average) of a list of numbers
   - Example: mean([10.5, 12.3, 8.7, 15.2, 9.8]) -> 11.3

2. median(numbers: List[float]) -> float
   - Description: Calculate the median (middle value) of a list of numbers
   - Example: median([10.5, 12.3, 8.7, 15.2, 9.8]) -> 10.5

3. mode(numbers: List[float]) -> List[float]
   - Description: Find the most frequent value(s) in a list of numbers
   - Example: mode([1, 2, 2, 3, 3, 3, 4, 5]) -> [3]

4. standard_deviation(numbers: List[float]) -> float
   - Description: Calculate the standard deviation of a list of numbers
   - Example: standard_deviation([10.5, 12.3, 8.7, 15.2, 9.8]) -> 2.43

5. probability_distribution(frequencies: List[float]) -> List[float]
   - Description: Calculate probability distribution from frequencies
   - Example: probability_distribution([10, 20, 30, 40]) -> [0.1, 0.2, 0.3, 0.4]

6. eigenvalues(matrix: List[List[float]]) -> Dict
   - Description: Calculate eigenvalues and eigenvectors of a matrix
   - Example: eigenvalues([[4, 2], [1, 3]]) -> {"eigenvalues": [5, 2], "eigenvectors": [[0.894, 0.447], [-0.447, 0.894]]}

HOW TO USE THESE TOOLS:
1. To perform mathematical operations, call the corresponding function with the appropriate parameters
2. The parameters should be in the format specified in the documentation
3. When analyzing data, you can use multiple tools in sequence for comprehensive analysis
"""
        return docs
    
    @staticmethod
    def parse_data_from_text(text: str) -> List[float]:
        """
        Extract numeric data from text.
        
        Args:
            text: Text containing numeric data
            
        Returns:
            List[float]: Extracted numeric data
        """
        # Look for common patterns like lists, arrays, or comma-separated values
        # Try JSON format first (e.g., [1, 2, 3])
        try:
            # Find anything that looks like a JSON array
            matches = re.findall(r'\[.*?\]', text)
            for match in matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                        return [float(x) for x in data]
                except:
                    pass
        except:
            pass
        
        # Try comma-separated values
        try:
            # Find sequences of comma-separated numbers
            matches = re.findall(r'(?:\d+(?:\.\d+)?,\s*)+\d+(?:\.\d+)?', text)
            if matches:
                data = [float(x.strip()) for x in matches[0].split(',') if x.strip()]
                if data:
                    return data
        except:
            pass
        
        # Try to find any numbers in the text
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if numbers:
            return [float(x) for x in numbers]
        
        # If all else fails, return an empty list
        return []
    
    @staticmethod
    def mean(numbers: List[float]) -> float:
        """
        Calculate the arithmetic mean of a list of numbers.
        
        Args:
            numbers: List of numbers
            
        Returns:
            float: Mean value
        """
        result = TypeScriptTools.mean(numbers)
        if "error" in result:
            raise ValueError(result["error"])
        return result.get("result", 0.0)
    
    @staticmethod
    def median(numbers: List[float]) -> float:
        """
        Calculate the median of a list of numbers.
        
        Args:
            numbers: List of numbers
            
        Returns:
            float: Median value
        """
        result = TypeScriptTools.median(numbers)
        if "error" in result:
            raise ValueError(result["error"])
        return result.get("result", 0.0)
    
    @staticmethod
    def mode(numbers: List[float]) -> List[float]:
        """
        Find the most frequent values in a list of numbers.
        
        Args:
            numbers: List of numbers
            
        Returns:
            List[float]: Mode values
        """
        result = TypeScriptTools.mode(numbers)
        if "error" in result:
            raise ValueError(result["error"])
        return result.get("result", [])
    
    @staticmethod
    def standard_deviation(numbers: List[float]) -> float:
        """
        Calculate the standard deviation of a list of numbers.
        
        Args:
            numbers: List of numbers
            
        Returns:
            float: Standard deviation
        """
        result = TypeScriptTools.standard_deviation(numbers)
        if "error" in result:
            raise ValueError(result["error"])
        return result.get("result", 0.0)
    
    @staticmethod
    def probability_distribution(frequencies: List[float]) -> List[float]:
        """
        Calculate probability distribution from frequencies.
        
        Args:
            frequencies: List of frequency values
            
        Returns:
            List[float]: Probability distribution
        """
        result = TypeScriptTools.probability_distribution(frequencies)
        if "error" in result:
            raise ValueError(result["error"])
        return result.get("result", [])
    
    @staticmethod
    def eigenvalues(matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Calculate eigenvalues and eigenvectors of a matrix.
        
        Args:
            matrix: Square matrix
            
        Returns:
            Dict[str, Any]: Eigenvalues and eigenvectors
        """
        result = TypeScriptTools.eigenvalues(matrix)
        if "error" in result:
            raise ValueError(result["error"])
        return result.get("result", {})
    
    @staticmethod
    def analyze_query(query: str, data: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Analyze a natural language query using the appropriate tools.
        
        Args:
            query: Natural language query
            data: Optional data to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Parse data from query if not provided
        if data is None:
            data = MathToolInterface.parse_data_from_text(query)
        
        if not data:
            return {"error": "No data found in query or provided"}
        
        results = {}
        tools_used = []
        
        # Determine which tools to use based on the query
        if re.search(r'\bmean\b|\baverage\b', query, re.IGNORECASE):
            try:
                results["mean"] = MathToolInterface.mean(data)
                tools_used.append("mean")
            except Exception as e:
                results["mean_error"] = str(e)
        
        if re.search(r'\bmedian\b|\bmiddle\b', query, re.IGNORECASE):
            try:
                results["median"] = MathToolInterface.median(data)
                tools_used.append("median")
            except Exception as e:
                results["median_error"] = str(e)
        
        if re.search(r'\bmode\b|\bmost\s+common\b|\bmost\s+frequent\b', query, re.IGNORECASE):
            try:
                results["mode"] = MathToolInterface.mode(data)
                tools_used.append("mode")
            except Exception as e:
                results["mode_error"] = str(e)
        
        if re.search(r'\bstandard\s+deviation\b|\bstd\b|\bdeviation\b', query, re.IGNORECASE):
            try:
                results["standard_deviation"] = MathToolInterface.standard_deviation(data)
                tools_used.append("standard_deviation")
            except Exception as e:
                results["standard_deviation_error"] = str(e)
        
        if re.search(r'\bprobability\b|\bdistribution\b', query, re.IGNORECASE):
            try:
                results["probability_distribution"] = MathToolInterface.probability_distribution(data)
                tools_used.append("probability_distribution")
            except Exception as e:
                results["probability_distribution_error"] = str(e)
        
        if re.search(r'\beigen\b|\beigenvalues\b|\beigenvectors\b', query, re.IGNORECASE):
            # Check if data is a matrix
            is_matrix = isinstance(data, list) and all(isinstance(row, list) for row in data)
            if is_matrix:
                try:
                    results["eigenvalues"] = MathToolInterface.eigenvalues(data)
                    tools_used.append("eigenvalues")
                except Exception as e:
                    results["eigenvalues_error"] = str(e)
        
        # If no specific tools were requested, perform common analysis
        if not tools_used:
            try:
                results["mean"] = MathToolInterface.mean(data)
                tools_used.append("mean")
            except Exception as e:
                results["mean_error"] = str(e)
            
            try:
                results["median"] = MathToolInterface.median(data)
                tools_used.append("median")
            except Exception as e:
                results["median_error"] = str(e)
            
            try:
                results["standard_deviation"] = MathToolInterface.standard_deviation(data)
                tools_used.append("standard_deviation")
            except Exception as e:
                results["standard_deviation_error"] = str(e)
        
        return {
            "results": results,
            "tools_used": tools_used,
            "data_points": len(data),
            "data_summary": {
                "first_few": data[:5],
                "length": len(data)
            }
        }


# Example of how an LLM might use this interface
def llm_example_usage():
    """Example of how an LLM might use the tool interface."""
    # Step 1: Get tool documentation
    tools_doc = MathToolInterface.get_tool_documentation()
    print("Available tools:")
    print(tools_doc)
    
    # Step 2: Parse a user query
    user_query = "What's the average and standard deviation of [10.5, 12.3, 8.7, 15.2, 9.8]?"
    print(f"\nUser query: {user_query}")
    
    # Step 3: Extract data from the query
    data = MathToolInterface.parse_data_from_text(user_query)
    print(f"Extracted data: {data}")
    
    # Step 4: Perform the analysis
    mean_value = MathToolInterface.mean(data)
    std_value = MathToolInterface.standard_deviation(data)
    
    # Step 5: Format and return the results
    print(f"\nResults:")
    print(f"Mean: {mean_value}")
    print(f"Standard Deviation: {std_value}")
    
    # Alternative: Use the analyze_query method
    print("\nUsing analyze_query:")
    results = MathToolInterface.analyze_query(user_query)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    llm_example_usage()