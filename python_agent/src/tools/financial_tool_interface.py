"""
Financial Tool Interface for LLMs

This module extends the mathematical tools interface with specific
financial analysis capabilities.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# Import the base tool interface
from python_agent.src.tools.llm_tool_interface import MathToolInterface
from python_agent.src.tools.test_tools import TypeScriptTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FinancialToolInterface")

class FinancialToolInterface(MathToolInterface):
    """
    Extended interface for financial analysis tools.
    
    This class inherits from MathToolInterface and adds financial-specific
    analysis capabilities like volatility and correlation calculations.
    """
    
    @staticmethod
    def get_financial_tool_documentation() -> str:
        """
        Get documentation for financial analysis tools.
        
        Returns:
            str: Tool documentation
        """
        docs = MathToolInterface.get_tool_documentation()
        
        financial_docs = """
ADDITIONAL FINANCIAL ANALYSIS TOOLS

1. calculate_volatility(prices: List[float]) -> float
   - Description: Calculate the volatility (standard deviation of returns) of a price series
   - Example: calculate_volatility([100, 102, 98, 103, 105]) -> 2.45

2. calculate_correlation(series1: List[float], series2: List[float]) -> float
   - Description: Calculate the Pearson correlation coefficient between two data series
   - Example: calculate_correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) -> -1.0

3. calculate_returns(prices: List[float]) -> List[float]
   - Description: Calculate percentage returns from a price series
   - Example: calculate_returns([100, 102, 98, 103]) -> [2.0, -3.92, 5.1]

4. analyze_price_series(prices: List[float]) -> Dict
   - Description: Comprehensive analysis of a price series including returns, volatility, and trends
   - Returns dictionary with various metrics and analyses
"""
        
        return docs + financial_docs
    
    @staticmethod
    def calculate_returns(prices: List[float]) -> List[float]:
        """
        Calculate percentage returns from a price series.
        
        Args:
            prices: List of price points
            
        Returns:
            List[float]: Percentage returns
        """
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            daily_return = ((prices[i] - prices[i-1]) / prices[i-1]) * 100  # in percentage
            returns.append(daily_return)
        
        return returns
    
    @staticmethod
    def calculate_volatility(prices: List[float], annualize: bool = False, trading_days: int = 252) -> float:
        """
        Calculate volatility (standard deviation of returns) of a price series.
        
        Args:
            prices: List of price points
            annualize: Whether to annualize the volatility
            trading_days: Number of trading days in a year (default 252)
            
        Returns:
            float: Volatility as a percentage
        """
        # Calculate returns
        returns = FinancialToolInterface.calculate_returns(prices)
        
        if not returns:
            return 0.0
        
        # Calculate standard deviation of returns
        volatility = MathToolInterface.standard_deviation(returns)
        
        # Annualize if requested
        if annualize and volatility > 0:
            volatility = volatility * (trading_days ** 0.5)
        
        return volatility
    
    @staticmethod
    def calculate_correlation(series1: List[float], series2: List[float] = None) -> float:
        """
        Calculate correlation between two data series.
        
        If only one series is provided, calculate autocorrelation (lag 1).
        
        Args:
            series1: First data series
            series2: Second data series (optional)
            
        Returns:
            float: Correlation coefficient (-1 to 1)
        """
        if series2 is None:
            # Calculate autocorrelation (lag 1)
            if len(series1) < 2:
                return 0.0
            series2 = series1[1:]
            series1 = series1[:-1]
        
        # Ensure equal length
        min_len = min(len(series1), len(series2))
        if min_len < 2:
            return 0.0
        
        series1 = series1[:min_len]
        series2 = series2[:min_len]
        
        n = len(series1)
        
        # Calculate means
        mean1 = sum(series1) / n
        mean2 = sum(series2) / n
        
        # Calculate covariance and standard deviations
        covariance = sum((series1[i] - mean1) * (series2[i] - mean2) for i in range(n)) / n
        std1 = MathToolInterface.standard_deviation(series1)
        std2 = MathToolInterface.standard_deviation(series2)
        
        # Calculate correlation coefficient
        if std1 > 0 and std2 > 0:
            correlation = covariance / (std1 * std2)
            return correlation
        return 0.0
    
    @staticmethod
    def analyze_price_series(prices: List[float]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a price series.
        
        Args:
            prices: List of price points
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        if len(prices) < 2:
            return {"error": "Need at least 2 price points for analysis"}
        
        # Calculate basic statistics
        mean = MathToolInterface.mean(prices)
        median = MathToolInterface.median(prices)
        std_dev = MathToolInterface.standard_deviation(prices)
        
        # Calculate returns
        returns = FinancialToolInterface.calculate_returns(prices)
        mean_return = MathToolInterface.mean(returns) if returns else 0.0
        
        # Calculate volatility
        volatility = FinancialToolInterface.calculate_volatility(prices)
        annualized_volatility = FinancialToolInterface.calculate_volatility(prices, annualize=True)
        
        # Calculate autocorrelation
        autocorrelation = FinancialToolInterface.calculate_correlation(prices)
        returns_autocorrelation = FinancialToolInterface.calculate_correlation(returns) if len(returns) > 1 else 0.0
        
        # Calculate trends
        trend = "Upward" if prices[-1] > prices[0] else "Downward" if prices[-1] < prices[0] else "Flat"
        price_change = ((prices[-1] - prices[0]) / prices[0]) * 100  # percentage change
        
        # Calculate max drawdown
        max_drawdown = 0.0
        peak = prices[0]
        for price in prices:
            if price > peak:
                peak = price
            else:
                drawdown = (peak - price) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        # Assemble results
        results = {
            "price_statistics": {
                "mean": mean,
                "median": median,
                "standard_deviation": std_dev,
                "min": min(prices),
                "max": max(prices),
                "range": max(prices) - min(prices)
            },
            "return_statistics": {
                "mean_return": mean_return,
                "volatility": volatility,
                "annualized_volatility": annualized_volatility,
                "min_return": min(returns) if returns else 0.0,
                "max_return": max(returns) if returns else 0.0
            },
            "trend_analysis": {
                "overall_trend": trend,
                "price_change_percent": price_change,
                "max_drawdown_percent": max_drawdown
            },
            "correlation_analysis": {
                "price_autocorrelation": autocorrelation,
                "returns_autocorrelation": returns_autocorrelation
            }
        }
        
        return results
    
    @staticmethod
    def analyze_financial_query(query: str, data: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Analyze a financial query using the appropriate tools.
        
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
            return {"error": "No financial data found in query or provided"}
        
        results = {}
        tools_used = []
        
        # Check for volatility-related queries
        if re.search(r'\bvolatility\b|\bstd\b|\bstandard deviation\b|\brisk\b', query, re.IGNORECASE):
            try:
                volatility = FinancialToolInterface.calculate_volatility(data)
                results["volatility"] = volatility
                tools_used.append("calculate_volatility")
                
                # If specifically asking for annualized
                if re.search(r'\bannualized\b|\bann\b|\byearly\b', query, re.IGNORECASE):
                    annualized_volatility = FinancialToolInterface.calculate_volatility(data, annualize=True)
                    results["annualized_volatility"] = annualized_volatility
            except Exception as e:
                results["volatility_error"] = str(e)
        
        # Check for correlation-related queries
        if re.search(r'\bcorrelation\b|\bcorrelated\b|\brelationship\b', query, re.IGNORECASE):
            try:
                autocorrelation = FinancialToolInterface.calculate_correlation(data)
                results["autocorrelation"] = autocorrelation
                tools_used.append("calculate_correlation")
            except Exception as e:
                results["correlation_error"] = str(e)
        
        # Check for returns-related queries
        if re.search(r'\breturns\b|\bperformance\b', query, re.IGNORECASE):
            try:
                returns = FinancialToolInterface.calculate_returns(data)
                mean_return = MathToolInterface.mean(returns) if returns else 0.0
                results["returns"] = returns[:5]  # First 5 returns only
                results["mean_return"] = mean_return
                tools_used.append("calculate_returns")
            except Exception as e:
                results["returns_error"] = str(e)
        
        # For comprehensive analysis
        if re.search(r'\banalyze\b|\banalysis\b|\bcomprehensive\b', query, re.IGNORECASE):
            try:
                analysis = FinancialToolInterface.analyze_price_series(data)
                results["comprehensive_analysis"] = analysis
                tools_used.append("analyze_price_series")
            except Exception as e:
                results["analysis_error"] = str(e)
        
        # If no specific financial tools were requested, perform basic analysis
        if not tools_used:
            try:
                volatility = FinancialToolInterface.calculate_volatility(data)
                results["volatility"] = volatility
                tools_used.append("calculate_volatility")
                
                if len(data) > 1:
                    autocorrelation = FinancialToolInterface.calculate_correlation(data)
                    results["autocorrelation"] = autocorrelation
                    tools_used.append("calculate_correlation")
            except Exception as e:
                results["basic_analysis_error"] = str(e)
        
        return {
            "results": results,
            "tools_used": tools_used,
            "data_points": len(data),
            "data_summary": {
                "first_few": data[:5],
                "length": len(data)
            }
        }

# Example usage
if __name__ == "__main__":
    # Sample price data
    prices = [100, 102, 98, 103, 105, 104, 107, 109, 108, 110]
    
    # Calculate volatility
    volatility = FinancialToolInterface.calculate_volatility(prices)
    print(f"Volatility: {volatility:.4f}%")
    
    # Calculate returns
    returns = FinancialToolInterface.calculate_returns(prices)
    print(f"Returns: {returns}")
    
    # Calculate correlation
    correlation = FinancialToolInterface.calculate_correlation(prices)
    print(f"Autocorrelation: {correlation:.4f}")
    
    # Comprehensive analysis
    analysis = FinancialToolInterface.analyze_price_series(prices)
    print("\nComprehensive Analysis:")
    print(json.dumps(analysis, indent=2))