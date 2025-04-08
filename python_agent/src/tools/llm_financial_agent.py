#!/usr/bin/env python
"""
LLM Financial Agent

This module provides a simple LLM-ready agent for financial analysis
that responds to natural language queries about financial data.
"""

import json
import re
import asyncio
import argparse
import logging
from typing import List, Dict, Any, Optional, Union

# Import the financial tool interface
from python_agent.src.tools.financial_tool_interface import FinancialToolInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLMFinancialAgent")

class LLMFinancialAgent:
    """
    Agent that responds to natural language queries about financial data.
    
    This agent provides a simple interface for LLMs to analyze financial data
    through natural language queries.
    """
    
    def __init__(self):
        """Initialize the LLM financial agent."""
        logger.info("LLM Financial Agent initialized")
    
    def get_capabilities(self) -> str:
        """
        Get a description of the agent's capabilities.
        
        Returns:
            str: Capability description
        """
        return """
I can analyze financial data through natural language queries. Here are my capabilities:

1. Calculate basic statistics (mean, median, standard deviation)
2. Calculate volatility of price series
3. Calculate correlations between data series
4. Analyze returns and performance metrics
5. Identify trends and patterns in financial data
6. Perform comprehensive financial analysis

To use these capabilities, provide a natural language query and financial data.
"""
    
    def load_data(self, file_path: str) -> List[float]:
        """
        Load financial data from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            List[float]: Financial data
        """
        try:
            with open(file_path, 'r') as f:
                # Try to parse as JSON first
                try:
                    json_data = json.load(f)
                    # Check if it's a run file with input_data
                    if isinstance(json_data, dict) and "input_data" in json_data:
                        return json_data["input_data"]
                    # Check if it's a simple list
                    elif isinstance(json_data, list):
                        return [float(x) for x in json_data]
                except json.JSONDecodeError:
                    # Try as CSV or plain text
                    content = f.read().strip()
                    return [float(x) for x in content.replace('\n', ',').split(',') if x.strip()]
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return []
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess a natural language query.
        
        Args:
            query: User query
            
        Returns:
            str: Preprocessed query
        """
        # Convert to lowercase for easier matching
        query = query.lower()
        
        # Expand common abbreviations
        query = query.replace("std", "standard deviation")
        query = query.replace("vol", "volatility")
        query = query.replace("corr", "correlation")
        
        return query
    
    def generate_response(self, query: str, data: List[float], verbose: bool = False) -> str:
        """
        Generate a natural language response to a financial query.
        
        Args:
            query: Preprocessed query
            data: Financial data
            verbose: Whether to include detailed analysis
            
        Returns:
            str: Natural language response
        """
        # Analyze the query and data
        analysis = FinancialToolInterface.analyze_financial_query(query, data)
        
        # Extract results
        results = analysis.get("results", {})
        tools_used = analysis.get("tools_used", [])
        
        # Generate appropriate response based on the query and results
        if "error" in analysis:
            return f"I couldn't analyze the data: {analysis['error']}"
        
        response = f"I analyzed {len(data)} financial data points"
        
        # Add specific responses based on tools used
        if "calculate_volatility" in tools_used and "volatility" in results:
            response += f"\n\nThe volatility of the price series is {results['volatility']:.4f}%, "
            
            if "annualized_volatility" in results:
                response += f"with an annualized volatility of {results['annualized_volatility']:.4f}%. "
            else:
                response += "which indicates the level of price fluctuation. "
                
            # Add interpretation
            if results['volatility'] < 1.0:
                response += "This represents relatively low volatility, suggesting stable prices."
            elif results['volatility'] < 3.0:
                response += "This represents moderate volatility, typical for many financial instruments."
            else:
                response += "This represents high volatility, indicating significant price fluctuations."
        
        if "calculate_correlation" in tools_used and "autocorrelation" in results:
            response += f"\n\nThe autocorrelation (lag 1) is {results['autocorrelation']:.4f}, "
            
            # Add interpretation
            if abs(results['autocorrelation']) < 0.2:
                response += "which suggests little to no correlation between consecutive prices."
            elif abs(results['autocorrelation']) < 0.5:
                response += "which indicates moderate correlation between consecutive prices."
            else:
                response += "which shows strong correlation between consecutive prices, suggesting potential trends or patterns."
        
        if "calculate_returns" in tools_used and "mean_return" in results:
            response += f"\n\nThe average return is {results['mean_return']:.4f}%, "
            
            # Add interpretation
            if results['mean_return'] > 0:
                response += "indicating an overall positive performance."
            else:
                response += "indicating an overall negative performance."
            
            if "returns" in results:
                response += f" Some example returns include: {', '.join(f'{r:.2f}%' for r in results['returns'])}."
        
        if "analyze_price_series" in tools_used and "comprehensive_analysis" in results:
            analysis = results["comprehensive_analysis"]
            
            # Add trend information
            trend_info = analysis.get("trend_analysis", {})
            if trend_info:
                response += f"\n\nThe overall trend is {trend_info.get('overall_trend', 'Unknown')}"
                if "price_change_percent" in trend_info:
                    response += f", with a {trend_info['price_change_percent']:.2f}% change in price"
                if "max_drawdown_percent" in trend_info:
                    response += f". The maximum drawdown was {trend_info['max_drawdown_percent']:.2f}%"
                response += "."
            
            # Add key statistics if verbose mode is enabled
            if verbose:
                price_stats = analysis.get("price_statistics", {})
                return_stats = analysis.get("return_statistics", {})
                
                response += "\n\nDetailed Statistics:"
                
                if price_stats:
                    response += "\n- Price Range: "
                    if "min" in price_stats and "max" in price_stats:
                        response += f"{price_stats['min']:.2f} to {price_stats['max']:.2f}"
                    
                    if "standard_deviation" in price_stats:
                        response += f"\n- Price Standard Deviation: {price_stats['standard_deviation']:.4f}"
                
                if return_stats:
                    if "volatility" in return_stats:
                        response += f"\n- Volatility: {return_stats['volatility']:.4f}%"
                    if "annualized_volatility" in return_stats:
                        response += f"\n- Annualized Volatility: {return_stats['annualized_volatility']:.4f}%"
        
        # If nothing specific was found, provide basic statistics
        if not tools_used or len(response.split('\n')) <= 1:
            response += "\n\nHere are some basic statistics about the data:"
            mean = FinancialToolInterface.mean(data)
            median = FinancialToolInterface.median(data)
            std_dev = FinancialToolInterface.standard_deviation(data)
            
            response += f"\n- Mean: {mean:.4f}"
            response += f"\n- Median: {median:.4f}"
            response += f"\n- Standard Deviation: {std_dev:.4f}"
        
        return response
    
    async def process_query(self, query: str, data: Optional[List[float]] = None, 
                      data_file: Optional[str] = None, verbose: bool = False) -> str:
        """
        Process a natural language query about financial data.
        
        Args:
            query: User query
            data: Optional list of data points
            data_file: Optional path to data file
            verbose: Whether to include detailed analysis
            
        Returns:
            str: Natural language response
        """
        # Load data if file provided and no data list
        if data is None and data_file:
            data = self.load_data(data_file)
        
        # If still no data, try to extract from query
        if not data:
            data = FinancialToolInterface.parse_data_from_text(query)
        
        # If still no data, return error
        if not data:
            return "I couldn't find any financial data to analyze. Please provide data directly or specify a data file."
        
        # Preprocess the query
        preprocessed_query = self.preprocess_query(query)
        
        # Generate response
        response = self.generate_response(preprocessed_query, data, verbose)
        
        return response

async def main():
    """Main function for command-line use."""
    parser = argparse.ArgumentParser(description="LLM Financial Agent")
    parser.add_argument("query", help="Natural language query about financial data")
    parser.add_argument("--data", help="Comma-separated data points")
    parser.add_argument("--file", help="Path to data file (JSON, CSV, or plain text)")
    parser.add_argument("--verbose", action="store_true", help="Include detailed analysis")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = LLMFinancialAgent()
    
    # Parse data if provided
    data = None
    if args.data:
        try:
            data = [float(x.strip()) for x in args.data.split(',') if x.strip()]
        except ValueError:
            print("Error parsing data. Please provide comma-separated numbers.")
            return
    
    # Process the query
    response = await agent.process_query(args.query, data, args.file, args.verbose)
    
    # Print the response
    print(response)

if __name__ == "__main__":
    asyncio.run(main())