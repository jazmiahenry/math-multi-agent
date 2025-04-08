"""
Enhanced Financial Analysis System with Tool Sequence Tracking

This module provides the entry point for running the financial analysis system,
integrating with the virtual tool manager and financial tools while providing
detailed tool sequence information and reasoning.
"""

import os
import asyncio
import argparse
import csv
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Import our existing modules
from python_agent.src.tools.financial_tool_interface import FinancialToolInterface
from python_agent.src.tools.llm_financial_agent import LLMFinancialAgent
from python_agent.src.tools.llm_tool_interface import MathToolInterface
import python_agent.src.tools.virtual_tool_manager as vtm
from python_agent.src.utils.logger import logger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/financial_analysis.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedFinancialAnalysis")

# Make sure logs directory exists
os.makedirs("logs", exist_ok=True)

class ToolUsageTracker:
    """
    Tracks and explains tool usage for better transparency.
    """
    
    @staticmethod
    def get_tool_reasoning(query: str, tool_name: str) -> str:
        """
        Generate reasoning for why a particular tool was chosen.
        
        Args:
            query: The user's query
            tool_name: The name of the tool used
            
        Returns:
            str: Reasoning explanation
        """
        query = query.lower()
        
        # Reasoning mappings for different tools
        reasoning_map = {
            "calculate_volatility": "chosen because the query mentions analyzing volatility or risk, which requires calculation of return fluctuations",
            "calculate_correlation": "chosen because the query asks to analyze relationships between data points or correlation patterns",
            "calculate_returns": "chosen to understand price changes over time by calculating percentage movements between consecutive data points",
            "analyze_price_series": "chosen to provide comprehensive analysis including volatility, trend, correlation, and return metrics",
            "mean": "chosen to calculate the average value, which provides a central tendency measurement",
            "median": "chosen to find the middle value, which is resistant to outliers unlike the mean",
            "mode": "chosen to identify the most frequently occurring value(s) in the dataset",
            "standard_deviation": "chosen to measure the amount of variation or dispersion in the dataset",
            "std_deviation": "chosen to measure the amount of variation or dispersion in the dataset",
            "probability_distribution": "chosen to calculate the relative likelihood of different values occurring",
            "eigenvalues": "chosen to perform matrix decomposition for advanced statistical analysis"
        }
        
        # Default reasoning if specific tool not found
        default_reasoning = f"chosen as the most appropriate tool based on the query pattern and data characteristics"
        
        # Check for specific keywords in the query to enhance reasoning
        if "volatility" in query and tool_name == "calculate_volatility":
            return "chosen because the query explicitly requests volatility calculation"
        
        if "correlation" in query and tool_name == "calculate_correlation":
            return "chosen because the query explicitly requests correlation analysis"
        
        if "comprehensive" in query and tool_name == "analyze_price_series":
            return "chosen to provide a full suite of analytics as requested in the query"
        
        # Return the specific reasoning or default
        return reasoning_map.get(tool_name, default_reasoning)
    
    @staticmethod
    def get_typescript_tool_mapping(tool_name: str) -> List[str]:
        """
        Map high-level tools to the sequence of TypeScript tools they use.
        
        Args:
            tool_name: The name of the high-level tool
            
        Returns:
            List[str]: List of TypeScript tools used in sequence
        """
        # Define tool sequences for financial tools
        tool_sequence_map = {
            "calculate_volatility": ["std_deviation"],
            "calculate_correlation": ["mean", "std_deviation"],
            "calculate_returns": [],  # Pure Python implementation
            "analyze_price_series": ["mean", "median", "std_deviation"],
            "mean": ["mean"],
            "median": ["median"],
            "mode": ["mode"],
            "standard_deviation": ["std_deviation"],
            "std_deviation": ["std_deviation"],
            "probability_distribution": ["probability"],
            "eigenvalues": ["eigen"]
        }
        
        return tool_sequence_map.get(tool_name, [])

def handle_simulated_error(error_msg, tool_name, data):
    """Handle simulated errors and provide alternate calculations."""
    # Log the error
    logger.error(f"Simulated error detected in tool: {tool_name} - {error_msg}")

    # Create fallback calculation based on the tool
    if tool_name == "mean":
        result = sum(data) / len(data) if data else 0
        solution = "Using Python's built-in sum and division for mean calculation"
    elif tool_name == "median":
        sorted_data = sorted(data)
        n = len(sorted_data)
        result = (sorted_data[n//2-1] + sorted_data[n//2]) / 2 if n % 2 == 0 else sorted_data[n//2]
        solution = "Using Python's built-in sorted function for median calculation"
    elif tool_name == "std_deviation" or tool_name == "standard_deviation":
        mean = sum(data) / len(data)
        result = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
        solution = "Using Python's built-in math operations for standard deviation"
    elif tool_name == "calculate_volatility":
        # Calculate returns first
        returns = []
        for i in range(1, len(data)):
            daily_return = ((data[i] - data[i-1]) / data[i-1]) * 100
            returns.append(daily_return)
        # Calculate std dev of returns
        mean_return = sum(returns) / len(returns)
        result = (sum((x - mean_return) ** 2 for x in returns) / len(returns)) ** 0.5
        solution = "Using Python implementation to calculate volatility from returns"
    elif tool_name == "calculate_correlation":
        series1 = data[:-1]
        series2 = data[1:]
        mean1 = sum(series1) / len(series1)
        mean2 = sum(series2) / len(series2)
        covariance = sum((series1[i] - mean1) * (series2[i] - mean2) for i in range(len(series1))) / len(series1)
        std1 = (sum((x - mean1) ** 2 for x in series1) / len(series1)) ** 0.5
        std2 = (sum((x - mean2) ** 2 for x in series2) / len(series2)) ** 0.5
        result = covariance / (std1 * std2) if std1 > 0 and std2 > 0 else 0
        solution = "Using Python implementation to calculate autocorrelation"
    else:
        result = None
        solution = f"No fallback calculation available for {tool_name}"

    return {
        "error_tool": tool_name,
        "error_message": error_msg,
        "fallback_result": result,
        "solution": solution
    }

async def process_query_with_tracking(query: str, data_source: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Process a financial query with enhanced tracking and explanations.
    
    Args:
        query: The user's query
        data_source: Path to data file or JSON string
        verbose: Include detailed analysis
        
    Returns:
        Dict[str, Any]: Enhanced results with tool tracking
    """
    # Initialize components
    agent = LLMFinancialAgent()
    vtool_manager = vtm.VirtualToolManager()
    
    # Register basic tools
    for tool_name in ["mean", "median", "mode", "std_deviation", "probability", "eigen"]:
        vtool_manager.register_tool(
            tool_name, 
            lambda params, name=tool_name: {"result": MathToolInterface.mean(params)}
                if name == "mean" else
            {"result": MathToolInterface.median(params)}
                if name == "median" else
            {"result": MathToolInterface.mode(params)}
                if name == "mode" else
            {"result": MathToolInterface.standard_deviation(params)}
                if name == "std_deviation" else
            {"result": MathToolInterface.probability_distribution(params)}
                if name == "probability" else
            {"result": MathToolInterface.eigenvalues(params)}
                if name == "eigen" else
            {"error": "Unknown tool"}
        )
    
    # Load data first
    try:
        data = agent.load_data(data_source)
        if not data:
            return {
                "error": "No data could be loaded from the source",
                "success": False
            }
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return {
            "error": f"Error loading data: {str(e)}",
            "success": False
        }
    
    # Process the query using LLMFinancialAgent
    response = await agent.process_query(query, data=data, verbose=verbose)
    
    # Get analysis results for detailed metrics
    analysis = FinancialToolInterface.analyze_financial_query(query, data)
    
    # Initialize tracking variables
    metrics = {}
    error_details = []
    
    # Extract the tools used
    tools_used = analysis.get("tools_used", [])
    
    # Check for simulated errors in the analysis results
    results = analysis.get("results", {})
    for tool_name, result in results.items():
        # Check for error strings
        if isinstance(result, str) and "Simulated error" in result:
            # Handle the error
            error_info = handle_simulated_error(result, tool_name, data)
            error_details.append(error_info)
            
            # Add fallback calculation to metrics
            metrics[tool_name + "_fallback"] = error_info["fallback_result"]
            metrics[tool_name + "_solution"] = error_info["solution"]
        elif isinstance(result, dict) and "error" in result and "Simulated error" in str(result["error"]):
            # Handle the error
            error_info = handle_simulated_error(result["error"], tool_name, data)
            error_details.append(error_info)
            
            # Add fallback calculation to metrics
            metrics[tool_name + "_fallback"] = error_info["fallback_result"]
            metrics[tool_name + "_solution"] = error_info["solution"]
        else:
            # No error, add the result to metrics
            metrics[tool_name] = result
    
    # Generate enhanced tool sequence with reasoning
    enhanced_tool_sequence = []
    for i, tool in enumerate(tools_used):
        # Map to TypeScript tools
        typescript_tools = ToolUsageTracker.get_typescript_tool_mapping(tool)
        
        # Generate reasoning
        reasoning = ToolUsageTracker.get_tool_reasoning(query, tool)
        
        # Create enhanced tool entry
        tool_entry = {
            "step_id": i+1,
            "tool": tool,
            "reasoning": reasoning,
            "typescript_tools": typescript_tools,
            "payload": {"data": data[:5]} if i == 0 else {}  # Only show sample data
        }
        
        # Check if this tool had an error
        had_error = False
        for error in error_details:
            if error["error_tool"] == tool:
                had_error = True
                tool_entry["error"] = error["error_message"]
                tool_entry["fallback_result"] = error["fallback_result"]
                tool_entry["solution"] = error["solution"]
                break
        
        tool_entry["status"] = "error" if had_error else "success"
        enhanced_tool_sequence.append(tool_entry)
    
    # Check if we have a matching virtual tool
    virtual_tool = vtool_manager.find_matching_tool(query, {"data": data})
    virtual_tool_info = None
    
    if virtual_tool:
        logger.info(f"Found matching virtual tool: {virtual_tool.name}")
        virtual_tool_info = {
            "name": virtual_tool.name,
            "description": virtual_tool.description,
            "confidence": virtual_tool.confidence,
            "success_rate": virtual_tool.success_rate,
            "execution_count": virtual_tool.execution_count
        }
    
    # Prepare comprehensive output
    output_results = {
        "task": query,
        "input_data": data[:10] + (["..."] if len(data) > 10 else []),  # First 10 data points
        "data_summary": {
            "length": len(data),
            "min": min(data) if data else None,
            "max": max(data) if data else None
        },
        "success": True,
        "task_answer": response,
        "metrics": metrics,
        "enhanced_tool_sequence": enhanced_tool_sequence,
        "virtual_tool_used": virtual_tool_info,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add error details if there were any
    if error_details:
        output_results["contains_simulated_errors"] = True
        output_results["error_details"] = error_details
    
    return output_results

async def try_create_virtual_tool(query: str, data: List[float], tool_sequence: List[Dict]) -> Dict[str, Any]:
    """
    Attempt to create a virtual tool based on a successful query.
    
    Args:
        query: The user's query
        data: The processed data
        tool_sequence: The sequence of tools used
        
    Returns:
        Dict[str, Any]: Information about the created tool or None
    """
    # Check if we have at least one tool and it was successful
    if not tool_sequence:
        return None
    
    # Extract keywords from the query
    keywords = [word.lower() for word in query.split() 
                if len(word) > 3 
                and word.lower() not in ("this", "that", "with", "from", "what", "analysis", 
                                         "analyze", "calculate", "find", "show", "get", "the", "and")]
    
    # Only proceed if we have at least one keyword
    if not keywords:
        return None
    
    # Initialize virtual tool manager
    vtool_manager = vtm.VirtualToolManager()
    
    # Clean up the tool sequence for the virtual tool
    clean_sequence = []
    for step in tool_sequence:
        clean_step = {
            "tool": step["tool"],
            "param_mapping": {
                "data": {"type": "input", "name": "data"},
                "numbers": {"type": "input", "name": "data"},
                "prices": {"type": "input", "name": "data"}
            }
        }
        clean_sequence.append(clean_step)
    
    # Create a virtual tool
    tool = vtool_manager.create_virtual_tool(
        name=f"{keywords[0].capitalize()}Analysis",
        description=f"Tool for {query}",
        tool_sequence=clean_sequence,
        problem_pattern={
            "keywords": keywords,
            "required_params": ["data"],
            "data_types": {"data": "list"}
        },
        reasoning=f"Created based on successful execution of query: {query}"
    )
    
    return {
        "name": tool.name,
        "description": tool.description,
        "keywords": keywords,
        "confidence": tool.confidence
    }

async def main():
    """Main entry point for the Enhanced Financial Analysis system."""
    parser = argparse.ArgumentParser(description="Enhanced Financial Analysis System")
    parser.add_argument("--query", type=str, required=True, 
                      help="Natural language query for analysis")
    parser.add_argument("--data", type=str, required=True,
                      help="Path to data file or comma-separated values")
    parser.add_argument("--output", type=str, default="output/enhanced_results.json", 
                      help="Path to output JSON file")
    parser.add_argument("--verbose", action="store_true", 
                      help="Include detailed analysis")
    parser.add_argument("--create-tool", action="store_true",
                      help="Attempt to create a virtual tool from this execution")
    
    args = parser.parse_args()
    
    try:
        # Process the query with enhanced tracking
        results = await process_query_with_tracking(args.query, args.data, args.verbose)
        
        # Check if the operation was successful
        if "error" in results and not results.get("success", False):
            print(f"Error: {results['error']}")
            return None
        
        # Try to create a virtual tool if requested
        if args.create_tool:
            tool_info = await try_create_virtual_tool(
                args.query, 
                results["input_data"], 
                results["enhanced_tool_sequence"]
            )
            
            if tool_info:
                results["created_virtual_tool"] = tool_info
                print(f"\nCreated new virtual tool: {tool_info['name']}")
        
        # Output to terminal
        print("\n--- Analysis Results ---")
        print(f"Query: {args.query}")
        print("\nResponse:")
        print(results["task_answer"])
        
        # Display any error details
        if "contains_simulated_errors" in results and results["contains_simulated_errors"]:
            print("\n⚠️ Simulated Errors Detected and Handled:")
            for error in results["error_details"]:
                print(f"  Tool: {error['error_tool']}")
                print(f"  Error: {error['error_message']}")
                print(f"  Fallback Result: {error['fallback_result']}")
                print(f"  Solution: {error['solution']}")
        
        print("\nTools Used:")
        for tool in results["enhanced_tool_sequence"]:
            status = "❌" if tool.get("status") == "error" else "✅"
            print(f"  {status} {tool['tool']}")
            print(f"    Reasoning: {tool['reasoning']}")
            if tool.get("typescript_tools"):
                print(f"    TypeScript Tools: {', '.join(tool['typescript_tools'])}")
            if tool.get("fallback_result") is not None:
                print(f"    Fallback Result: {tool['fallback_result']}")
                print(f"    Solution: {tool['solution']}")
        
        # Create directory for output if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save to JSON file
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    asyncio.run(main())