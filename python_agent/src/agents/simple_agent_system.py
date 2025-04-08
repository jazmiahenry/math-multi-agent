"""
Simple Agent System for Mathematical Analysis

This module provides a simplified interface for LLMs to use TypeScript
mathematical tools without the complexity of reinforcement learning.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleAgentSystem")

# Import the TypeScriptTools class
from python_agent.src.tools.test_tools import TypeScriptTools, call_math_tool

class SimpleAgentSystem:
    """
    A simplified agent system that provides direct access to mathematical tools.
    
    This class removes the reinforcement learning components and complex planning
    to provide a streamlined interface for LLMs to perform mathematical operations.
    """
    
    def __init__(self):
        """Initialize the simple agent system."""
        self.available_tools = {
            "mean": self._mean,
            "median": self._median,
            "mode": self._mode,
            "standard_deviation": self._standard_deviation,
            "probability_distribution": self._probability_distribution,
            "eigenvalues": self._eigenvalues,
        }
        logger.info("Simple Agent System initialized")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get information about available tools for LLM use.
        
        Returns:
            List[Dict[str, Any]]: List of tool definitions
        """
        tools = []
        
        # Add mean tool
        tools.append({
            "name": "mean",
            "description": "Calculate the arithmetic mean of a list of numbers",
            "parameters": {
                "numbers": "List of numbers to calculate the mean"
            },
            "example": "mean([10.5, 12.3, 8.7, 15.2, 9.8])"
        })
        
        # Add median tool
        tools.append({
            "name": "median",
            "description": "Calculate the median of a list of numbers",
            "parameters": {
                "numbers": "List of numbers to calculate the median"
            },
            "example": "median([10.5, 12.3, 8.7, 15.2, 9.8])"
        })
        
        # Add mode tool
        tools.append({
            "name": "mode",
            "description": "Find the most frequent values in a list of numbers",
            "parameters": {
                "numbers": "List of numbers to find the mode"
            },
            "example": "mode([1, 2, 2, 3, 3, 3, 4, 5])"
        })
        
        # Add standard deviation tool
        tools.append({
            "name": "standard_deviation",
            "description": "Calculate the standard deviation of a list of numbers",
            "parameters": {
                "numbers": "List of numbers to calculate the standard deviation"
            },
            "example": "standard_deviation([10.5, 12.3, 8.7, 15.2, 9.8])"
        })
        
        # Add probability distribution tool
        tools.append({
            "name": "probability_distribution",
            "description": "Calculate probability distribution from frequencies",
            "parameters": {
                "frequencies": "List of frequency values"
            },
            "example": "probability_distribution([10, 20, 30, 40])"
        })
        
        # Add eigenvalues tool
        tools.append({
            "name": "eigenvalues",
            "description": "Calculate eigenvalues and eigenvectors of a matrix",
            "parameters": {
                "matrix": "Square matrix in the form [[row1], [row2], ...]"
            },
            "example": "eigenvalues([[4, 2], [1, 3]])"
        })
        
        return tools
    
    async def execute_tool(self, tool_name: str, parameters: Any) -> Dict[str, Any]:
        """
        Execute a mathematical tool by name with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            
        Returns:
            Dict[str, Any]: Result of the tool execution
        """
        if tool_name not in self.available_tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            result = await self.available_tools[tool_name](parameters)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {"error": str(e)}
    
    async def _mean(self, numbers: List[float]) -> Dict[str, Any]:
        """Calculate the mean of a list of numbers."""
        return TypeScriptTools.mean(numbers)
    
    async def _median(self, numbers: List[float]) -> Dict[str, Any]:
        """Calculate the median of a list of numbers."""
        return TypeScriptTools.median(numbers)
    
    async def _mode(self, numbers: List[float]) -> Dict[str, Any]:
        """Calculate the mode of a list of numbers."""
        return TypeScriptTools.mode(numbers)
    
    async def _standard_deviation(self, numbers: List[float]) -> Dict[str, Any]:
        """Calculate the standard deviation of a list of numbers."""
        return TypeScriptTools.standard_deviation(numbers)
    
    async def _probability_distribution(self, frequencies: List[float]) -> Dict[str, Any]:
        """Calculate the probability distribution from frequencies."""
        return TypeScriptTools.probability_distribution(frequencies)
    
    async def _eigenvalues(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """Calculate eigenvalues and eigenvectors of a matrix."""
        return TypeScriptTools.eigenvalues(matrix)
    
    async def analyze_query(self, query: str, data: Any = None) -> Dict[str, Any]:
        """
        Analyze a natural language query and execute the appropriate tools.
        
        Args:
            query: Natural language query about data analysis
            data: Optional data to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # This is a simple rule-based approach - in a full implementation, you
        # would use an LLM to parse the query and determine which tools to call
        
        results = {"query": query, "results": {}, "tools_used": []}
        
        # Convert data to a list of floats if it's not already
        if data is None:
            # Extract numbers from the query if no data provided
            import re
            data = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", query)]
        
        if isinstance(data, str):
            try:
                # Try to parse as a comma-separated list
                data = [float(x.strip()) for x in data.split(",") if x.strip()]
            except:
                return {"error": "Could not parse data string into numbers"}
        
        if not data:
            return {"error": "No data provided or found in query"}
        
        # Execute basic statistics by default
        if "mean" in query.lower() or "average" in query.lower():
            mean_result = await self._mean(data)
            results["results"]["mean"] = mean_result.get("result", mean_result)
            results["tools_used"].append("mean")
        
        if "median" in query.lower():
            median_result = await self._median(data)
            results["results"]["median"] = median_result.get("result", median_result)
            results["tools_used"].append("median")
        
        if "mode" in query.lower() or "most frequent" in query.lower():
            mode_result = await self._mode(data)
            results["results"]["mode"] = mode_result.get("result", mode_result)
            results["tools_used"].append("mode")
        
        if "standard deviation" in query.lower() or "std" in query.lower() or "deviation" in query.lower():
            std_result = await self._standard_deviation(data)
            results["results"]["standard_deviation"] = std_result.get("result", std_result)
            results["tools_used"].append("standard_deviation")
        
        if "probability" in query.lower() or "distribution" in query.lower():
            prob_result = await self._probability_distribution(data)
            results["results"]["probability_distribution"] = prob_result.get("result", prob_result)
            results["tools_used"].append("probability_distribution")
        
        if "eigen" in query.lower() and isinstance(data, list) and all(isinstance(row, list) for row in data):
            eigen_result = await self._eigenvalues(data)
            results["results"]["eigenvalues"] = eigen_result.get("result", eigen_result)
            results["tools_used"].append("eigenvalues")
        
        # If no specific tools were requested, provide common statistics
        if not results["tools_used"]:
            mean_result = await self._mean(data)
            median_result = await self._median(data)
            std_result = await self._standard_deviation(data)
            
            results["results"]["mean"] = mean_result.get("result", mean_result)
            results["results"]["median"] = median_result.get("result", median_result)
            results["results"]["standard_deviation"] = std_result.get("result", std_result)
            
            results["tools_used"] = ["mean", "median", "standard_deviation"]
        
        # Add a simple summary
        tools_used = ", ".join(results["tools_used"])
        results["summary"] = f"Analysis completed using {tools_used}."
        
        return results


async def analyze_data(query: str, data: Any = None) -> Dict[str, Any]:
    """
    Analyze data using the simplified agent system.
    
    Args:
        query: Analysis query
        data: Data to analyze
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    system = SimpleAgentSystem()
    return await system.analyze_query(query, data)


# Example usage
if __name__ == "__main__":
    async def run_example():
        data = [10.5, 12.3, 8.7, 15.2, 9.8, 11.5, 14.2, 10.9, 13.1, 9.5]
        query = "Calculate the mean, median, and standard deviation of this data"
        
        results = await analyze_data(query, data)
        print(json.dumps(results, indent=2))
    
    asyncio.run(run_example())