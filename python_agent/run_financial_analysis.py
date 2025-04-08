#!/usr/bin/env python
"""
Run Financial Analysis Script

This script runs a volatility and correlation analysis on the data from run1.json
and outputs the results to ensure the tools are working correctly.
"""

import json
import sys
import os
import asyncio
from typing import List, Dict, Any
from datetime import datetime

# Import the financial tool interface
from python_agent.src.tools.financial_tool_interface import FinancialToolInterface

def load_run_data(filepath: str = "run1.json") -> tuple:
    """
    Load task and financial data from a run JSON file.
    
    Returns:
        tuple: (task, data)
    """
    try:
        with open(filepath, 'r') as f:
            run_data = json.load(f)
        
        # Extract the task and input data
        task = run_data.get("task", "")
        data = run_data.get("input_data", [])
        
        return task, data
    except Exception as e:
        print(f"Error loading run data: {e}")
        return "", []

async def analyze_financial_data(task: str, data: List[float]) -> Dict[str, Any]:
    """
    Analyze financial data using the financial tools.
    
    Args:
        task: Task description
        data: Financial data points
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    print(f"Analyzing task: {task}")
    print(f"Data points: {len(data)}")
    
    # Use the financial tool interface to analyze the data
    results = FinancialToolInterface.analyze_financial_query(task, data)
    
    return results

def save_results(results: Dict[str, Any], output_path: str = "analysis_results.json") -> None:
    """
    Save analysis results to a JSON file.
    
    Args:
        results: Analysis results
        output_path: Output file path
    """
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_path}")

async def main():
    """Main function."""
    # Set default filepath
    run_filepath = "run1.json"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        run_filepath = sys.argv[1]
    
    # Load data
    task, data = load_run_data(run_filepath)
    
    if not task or not data:
        print("Error: Missing task or data in the run file.")
        return
    
    try:
        # Run the analysis
        results = await analyze_financial_data(task, data)
        
        # Format and display results
        print("\n=== Analysis Results ===")
        
        if "results" in results:
            # Display volatility
            if "volatility" in results["results"]:
                print(f"Volatility: {results['results']['volatility']:.4f}%")
            
            # Display autocorrelation
            if "autocorrelation" in results["results"]:
                print(f"Autocorrelation: {results['results']['autocorrelation']:.4f}")
                
            # Display comprehensive analysis if available
            if "comprehensive_analysis" in results["results"]:
                analysis = results["results"]["comprehensive_analysis"]
                
                print("\nPrice Statistics:")
                for key, value in analysis["price_statistics"].items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                
                print("\nReturn Statistics:")
                for key, value in analysis["return_statistics"].items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                
                print("\nTrend Analysis:")
                for key, value in analysis["trend_analysis"].items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                
                print("\nCorrelation Analysis:")
                for key, value in analysis["correlation_analysis"].items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        
        print(f"\nTools used: {', '.join(results.get('tools_used', []))}")
        
        # Save results
        save_results(results)
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    asyncio.run(main())