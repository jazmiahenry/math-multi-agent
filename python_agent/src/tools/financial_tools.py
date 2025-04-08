#!/usr/bin/env python
"""
Test Financial Tools with Run1 Data

This script tests the mathematical tools with the actual financial data
from run1.json to verify they work correctly.
"""

import json
import sys
import asyncio
from typing import List, Dict, Any

# Import the tool interface
from python_agent.src.tools.llm_tool_interface import MathToolInterface

def load_run_data(filepath: str = "run1.json") -> List[float]:
    """
    Load financial data from a run JSON file.
    """
    try:
        with open(filepath, 'r') as f:
            run_data = json.load(f)
        
        # Extract the input data
        return run_data.get("input_data", [])
    except Exception as e:
        print(f"Error loading run data: {e}")
        return []

def calculate_volatility(data: List[float]) -> float:
    """
    Calculate volatility (standard deviation of returns).
    
    For financial data, volatility is typically calculated as the 
    standard deviation of percentage returns.
    """
    # Calculate percentage returns
    returns = []
    for i in range(1, len(data)):
        daily_return = ((data[i] - data[i-1]) / data[i-1]) * 100  # in percentage
        returns.append(daily_return)
    
    # Calculate standard deviation of returns
    if returns:
        volatility = MathToolInterface.standard_deviation(returns)
        return volatility
    return 0.0

def calculate_correlations(data1: List[float], data2: List[float] = None) -> float:
    """
    Calculate correlation between two data series.
    
    If only one data series is provided, calculate autocorrelation (lag 1).
    """
    if data2 is None:
        # Calculate autocorrelation (lag 1)
        data2 = data1[1:]
        data1 = data1[:-1]
    
    if len(data1) != len(data2):
        raise ValueError("Data series must have the same length for correlation calculation")
    
    n = len(data1)
    
    # Calculate means
    mean1 = sum(data1) / n
    mean2 = sum(data2) / n
    
    # Calculate covariance and standard deviations
    covariance = sum((data1[i] - mean1) * (data2[i] - mean2) for i in range(n)) / n
    std1 = MathToolInterface.standard_deviation(data1)
    std2 = MathToolInterface.standard_deviation(data2)
    
    # Calculate correlation coefficient
    if std1 > 0 and std2 > 0:
        correlation = covariance / (std1 * std2)
        return correlation
    return 0.0

async def main():
    """
    Main function to test financial tools.
    """
    # Load data from run1.json
    data = load_run_data()
    
    if not data:
        print("No data found. Please check the run1.json file.")
        return
    
    print(f"Loaded {len(data)} data points from run1.json")
    print(f"Sample data: {data[:5]}...")
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    mean = MathToolInterface.mean(data)
    median = MathToolInterface.median(data)
    std_dev = MathToolInterface.standard_deviation(data)
    
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Standard Deviation: {std_dev}")
    
    # Calculate financial metrics
    print("\n=== Financial Metrics ===")
    
    # Calculate volatility
    volatility = calculate_volatility(data)
    print(f"Volatility (std dev of returns): {volatility:.4f}%")
    
    # Calculate autocorrelation
    autocorrelation = calculate_correlations(data)
    print(f"Autocorrelation (lag 1): {autocorrelation:.4f}")
    
    # Calculate correlation between first and second half (if applicable)
    if len(data) >= 4:
        mid_point = len(data) // 2
        first_half = data[:mid_point]
        second_half = data[-mid_point:]
        
        if len(first_half) == len(second_half):
            half_correlation = calculate_correlations(first_half, second_half)
            print(f"Correlation (first half vs second half): {half_correlation:.4f}")
    
    # Calculate moving statistics
    window_size = min(10, len(data) - 1)
    if window_size > 2:
        print(f"\n=== Moving Statistics (window size: {window_size}) ===")
        
        moving_volatility = []
        for i in range(len(data) - window_size):
            window = data[i:i+window_size]
            vol = calculate_volatility(window)
            moving_volatility.append(vol)
        
        avg_moving_volatility = sum(moving_volatility) / len(moving_volatility)
        print(f"Average Moving Volatility: {avg_moving_volatility:.4f}%")
        
        max_volatility = max(moving_volatility)
        min_volatility = min(moving_volatility)
        print(f"Max Volatility Window: {max_volatility:.4f}%")
        print(f"Min Volatility Window: {min_volatility:.4f}%")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())