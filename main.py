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
        logging.FileHandler("financial_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedFinancialAnalysis")

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
    
    # Process the query using LLMFinancialAgent
    response = await agent.process_query(query, data_file=data_source, verbose=verbose)
    
    # Get analysis results for detailed metrics
    data = agent.load_data(data_source)
    analysis = FinancialToolInterface.analyze_financial_query(query, data)
    
    # Extract the tools used
    tools_used = analysis.get("tools_used", [])
    
    # Generate enhanced tool sequence with reasoning
    enhanced_tool_sequence = []
    for i, tool in enumerate(tools_used):
        # Map to TypeScript tools
        typescript_tools = ToolUsageTracker.get_typescript_tool_mapping(tool)
        
        # Generate reasoning
        reasoning = ToolUsageTracker.get_tool_reasoning(query, tool)
        
        # Create enhanced tool entry
        enhanced_tool = {
            "step_id": i+1,
            "tool": tool,
            "reasoning": reasoning,
            "typescript_tools": typescript_tools,
            "payload": {"data": data[:5]} if i == 0 else {}  # Only show sample data
        }
        
        enhanced_tool_sequence.append(enhanced_tool)
    
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
        "metrics": analysis.get("results", {}),
        "enhanced_tool_sequence": enhanced_tool_sequence,
        "virtual_tool_used": virtual_tool_info,
        "timestamp": datetime.now().isoformat()
    }
    
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
        
        print("\nTools Used:")
        for tool in results["enhanced_tool_sequence"]:
            print(f"  - {tool['tool']}")
            print(f"    Reasoning: {tool['reasoning']}")
            if tool["typescript_tools"]:
                print(f"    TypeScript Tools: {', '.join(tool['typescript_tools'])}")
        
        # Save to JSON file
        output_path = args.output
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {output_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    asyncio.run(main())