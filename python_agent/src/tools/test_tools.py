"""
TypeScript Tools Wrapper for Python

Provides a simple interface to call TypeScript mathematical tools via HTTP.
"""

import requests
import json
from typing import Union, List, Dict, Any

class TypeScriptTools:
    """
    Wrapper class for calling TypeScript mathematical tools.
    
    Methods provide a consistent interface for calling different statistical tools.
    """
    
    BASE_URL = "http://localhost:3000/tool"
    
    @classmethod
    def _call_tool(cls, tool_name: str, data: Union[Dict, List], key: str = "numbers") -> Dict[str, Any]:
        """
        Generic method to call a TypeScript tool.
        
        Args:
            tool_name: Name of the tool to call
            data: Input data for the tool
            key: Key to use for payload (default 'numbers')
        
        Returns:
            Dict containing the result or error
        """
        try:
            # Prepare the payload with the specified key
            payload = {key: data} if key != "matrix" else data
            
            # Make the request
            response = requests.post(
                f"{cls.BASE_URL}/{tool_name}", 
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            # If response is not successful, try to get error details
            if not response.ok:
                return {"error": f"{response.status_code} {response.reason}: {response.text}"}
            
            # Return the result
            return response.json()
        
        except requests.RequestException as e:
            return {"error": str(e)}
    
    @classmethod
    def mean(cls, numbers: List[float]) -> Dict[str, Union[float, str]]:
        """
        Calculate the mean of a list of numbers.
        """
        return cls._call_tool("mean", numbers)
    
    @classmethod
    def median(cls, numbers: List[float]) -> Dict[str, Union[float, str]]:
        """
        Calculate the median of a list of numbers.
        """
        return cls._call_tool("median", numbers)
    
    @classmethod
    def mode(cls, numbers: List[float]) -> Dict[str, Union[List[float], str]]:
        """
        Calculate the mode of a list of numbers.
        """
        return cls._call_tool("mode", numbers)
    
    @classmethod
    def standard_deviation(cls, numbers: List[float]) -> Dict[str, Union[float, str]]:
        """
        Calculate the standard deviation of a list of numbers.
        """
        return cls._call_tool("std_deviation", numbers)
    
    @classmethod
    def probability_distribution(cls, frequencies: List[float]) -> Dict[str, Union[List[float], str]]:
        """
        Calculate the probability distribution.
        """
        return cls._call_tool("probability", frequencies, key="frequencies")
    
    @classmethod
    def eigenvalues(cls, matrix: List[List[float]]) -> Dict[str, Union[Dict, str]]:
        """
        Calculate eigenvalues and eigenvectors.
        """
        return cls._call_tool("eigen", matrix, key="matrix")

def call_math_tool(tool_name, data):
    """
    Legacy function to maintain compatibility with existing code.
    
    Args:
        tool_name: Name of the mathematical tool
        data: Input data for the tool
    
    Returns:
        Dict with 'result' or 'error' key
    """
    try:
        # Use TypeScriptTools to handle different tool calls
        if tool_name == "mean":
            result = TypeScriptTools.mean(data)
        elif tool_name == "median":
            result = TypeScriptTools.median(data)
        elif tool_name == "mode":
            result = TypeScriptTools.mode(data)
        elif tool_name == "std_deviation":
            result = TypeScriptTools.standard_deviation(data)
        elif tool_name == "probability":
            result = TypeScriptTools.probability_distribution(data)
        elif tool_name == "eigen":
            result = TypeScriptTools.eigenvalues(data)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
        
        # If there's an error, return it
        if "error" in result:
            return result
        
        # Return the result
        return {"result": result.get("result", result)}
    
    except Exception as e:
        return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    # Demonstrate usage of different tools
    print("Mean:", TypeScriptTools.mean([10.5, 12.3, 8.7, 15.2, 9.8]))
    print("Median:", TypeScriptTools.median([10.5, 12.3, 8.7, 15.2, 9.8]))
    print("Mode:", TypeScriptTools.mode([1, 2, 2, 3, 3, 3, 4, 5]))
    print("Standard Deviation:", TypeScriptTools.standard_deviation([10.5, 12.3, 8.7, 15.2, 9.8]))
    print("Probability Distribution:", TypeScriptTools.probability_distribution([10, 20, 30, 40]))
    print("Eigenvalues:", TypeScriptTools.eigenvalues([[4, 2], [1, 3]]))