"""
Financial Analysis Multi-Agent System - Main Module

This module provides the entry point for the enhanced financial analysis multi-agent system.
It can process financial data from various sources (DataFrame, CSV, TXT, XLSX)
and coordinate the agents to perform the requested analysis with error detection
and virtual tool learning capabilities.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import autogen
from typing import Union, Dict, List, Optional, Any
import asyncio
import logging

# Import the enhanced agent system
# In a real implementation, you would use proper package imports
from python_agent.integration_module import EnhancedFinancialSystem
from python_agent.virtual_tool_manager import VirtualToolManager
from python_agent.test_tools import call_math_tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("financial_agents.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FinancialMain")


class FinancialDataProcessor:
    """
    Processes financial data from various sources and prepares it for analysis.
    
    This class handles loading and preprocessing financial data from different
    file formats and data structures.
    """
    
    @staticmethod
    def load_data(
        data_source: Union[str, pd.DataFrame], 
        sheet_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load financial data from various sources.
        
        Args:
            data_source: Path to a file (CSV, TXT, XLSX) or pandas DataFrame
            sheet_name: Sheet name for Excel files (optional)
            
        Returns:
            pd.DataFrame: Loaded and preprocessed financial data
        """
        if isinstance(data_source, pd.DataFrame):
            return data_source
        
        if not isinstance(data_source, str):
            raise ValueError("Data source must be a file path or pandas DataFrame")
        
        if not os.path.exists(data_source):
            raise FileNotFoundError(f"File not found: {data_source}")
        
        file_ext = os.path.splitext(data_source)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(data_source)
        elif file_ext == '.txt':
            # Try different delimiters for text files
            for delimiter in [',', '\t', '|', ';', ' ']:
                try:
                    df = pd.read_csv(data_source, delimiter=delimiter)
                    # If we got more than one column, assume this delimiter worked
                    if len(df.columns) > 1:
                        return df
                except Exception:
                    continue
            
            # Fallback to single column if no delimiter works
            return pd.read_csv(data_source, header=None, names=['value'])
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(data_source, sheet_name=sheet_name)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    @staticmethod
    def extract_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and convert numeric columns from a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with only numeric columns
        """
        # Try to convert columns to numeric, keep only those that succeed
        numeric_df = df.copy()
        
        for col in numeric_df.columns:
            try:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
            except Exception:
                # Drop columns that can't be converted to numeric
                numeric_df = numeric_df.drop(columns=[col])
        
        # Drop columns with too many NaN values (>50%)
        numeric_df = numeric_df.dropna(axis=1, thresh=len(numeric_df) // 2)
        
        if numeric_df.empty or len(numeric_df.columns) == 0:
            raise ValueError("No numeric data found in the provided dataset")
        
        return numeric_df
    
    @staticmethod
    def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the dataframe for the agents.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict: Summary statistics and information
        """
        numeric_df = FinancialDataProcessor.extract_numeric_columns(df)
        
        # Create data summary
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "numeric_columns": list(numeric_df.columns),
            "sample_rows": json.loads(df.head(5).to_json(orient="records")),
            "basic_stats": {}
        }
        
        # Generate basic statistics for each numeric column
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna().values.tolist()
            if len(col_data) > 0:
                summary["basic_stats"][col] = {
                    "min": float(numeric_df[col].min()),
                    "max": float(numeric_df[col].max()),
                    "mean": float(numeric_df[col].mean()),
                    "median": float(numeric_df[col].median()),
                    "std": float(numeric_df[col].std()),
                    "sample_values": col_data[:20]  # Limit to 20 values for the summary
                }
        
        return summary


class EnhancedFinancialAnalysisSystem:
    """
    Main class for running the enhanced financial analysis multi-agent system.
    
    This class coordinates the data processing and agent interaction
    to perform financial analysis on the provided data with error detection
    and virtual tool learning capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the enhanced financial analysis system.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration if provided
        self.config = self._load_config(config_path)
        
        # Initialize the enhanced financial system with error detection and learning
        self.system = EnhancedFinancialSystem(config_path)
        
        # Data processor for handling various data formats
        self.data_processor = FinancialDataProcessor()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict: Configuration dictionary
        """
        default_config = {
            "virtual_tool_storage": ".virtual_tools",
            "cache_dir": ".agent_cache",
            "log_dir": "./logs",
            "llm_config": {
                "model": "gpt-4",
                "temperature": 0.7,
                "config_list": "OAI_CONFIG_LIST"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
                return default_config
        
        return default_config
    
    async def run_analysis(
        self, 
        data_source: Union[str, pd.DataFrame],
        task: str,
        sheet_name: Optional[str] = None,
        human_input_mode: str = "TERMINATE"
    ) -> Dict:
        """
        Run a financial analysis task on the provided data.
        
        Args:
            data_source: Path to data file or pandas DataFrame
            task: Description of the analysis task to perform
            sheet_name: Sheet name for Excel files (optional)
            human_input_mode: Mode for human interaction ("TERMINATE", "ALWAYS", etc.)
            
        Returns:
            Dict: Results of the analysis
        """
        # Load and process the data
        try:
            df = self.data_processor.load_data(data_source, sheet_name)
            data_summary = self.data_processor.summarize_dataframe(df)
        except Exception as e:
            return {"error": f"Data loading error: {str(e)}"}
        
        # Create a user proxy agent for interaction
        user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=10,
            system_message="You are a financial analyst who needs help analyzing data."
        )
        
        # Set up a group chat for all agents
        # Include all agents from the enhanced system
        groupchat = autogen.GroupChat(
            agents=[user_proxy, self.system.planner, self.system.executor, self.system.learning_agent] + self.system.workers,
            messages=[],
            max_round=50
        )
        
        manager = autogen.GroupChatManager(groupchat=groupchat)
        
        # Construct a data-aware task message
        enhanced_task = f"""
I need to analyze the following financial data:

Data Summary:
- Dimensions: {data_summary['shape'][0]} rows × {data_summary['shape'][1]} columns
- Columns: {', '.join(data_summary['columns'])}
- Numeric columns: {', '.join(data_summary['numeric_columns'])}

Sample data (first 5 rows):
{json.dumps(data_summary['sample_rows'], indent=2)}

Basic statistics for key columns:
{json.dumps({k: v for k, v in data_summary['basic_stats'].items() if k in data_summary['numeric_columns'][:3]}, indent=2)}

Task: {task}

Generate a comprehensive analysis plan and execute it using the appropriate mathematical tools.
Be aware that some mathematical tools may occasionally produce incorrect results, so use verification
strategies to ensure accurate analysis.
"""
        
        # Initiate the conversation
        conversation_result = await user_proxy.initiate_chat(
            manager,
            message=enhanced_task
        )
        
        # Check for potential virtual tools based on this execution
        patterns = await self.system.analyze_for_virtual_tools()
        
        # Get statistics about virtual tools
        tool_stats = self.system.get_virtual_tool_stats()
        
        # Extract and format the results for return
        results = {
            "task": task,
            "data_source": str(data_source) if isinstance(data_source, str) else "DataFrame",
            "conversation": conversation_result,
            "virtual_tool_candidates": len(patterns) if patterns else 0,
            "virtual_tool_stats": tool_stats
        }
        
        return results
    
    def get_virtual_tool_info(self) -> Dict:
        """
        Get information about available virtual tools.
        
        Returns:
            Dict: Information about virtual tools
        """
        all_tools = self.system.virtual_tool_manager.get_all_virtual_tools()
        
        tool_info = []
        for tool in all_tools:
            tool_info.append({
                "name": tool.name,
                "description": tool.description,
                "confidence": tool.confidence,
                "execution_count": tool.execution_count,
                "success_rate": tool.success_rate,
                "created_at": tool.created_at
            })
        
        return {
            "count": len(tool_info),
            "tools": tool_info,
            "stats": self.system.get_virtual_tool_stats()
        }


def parse_arguments():
    """
    Parse command line arguments for the financial analysis system.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Enhanced Financial Analysis Multi-Agent System")
    
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Path to financial data file (CSV, TXT, XLSX)"
    )
    
    parser.add_argument(
        "--task", 
        type=str, 
        required=True,
        help="Description of the analysis task"
    )
    
    parser.add_argument(
        "--sheet", 
        type=str, 
        default=None,
        help="Sheet name for Excel files"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Enable interactive mode with human input"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Path to save the analysis results (JSON format)"
    )
    
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Show information about available virtual tools"
    )
    
    return parser.parse_args()


async def main_async():
    """
    Asynchronous main entry point for the financial analysis system.
    """
    args = parse_arguments()
    
    # Initialize the enhanced analysis system
    system = EnhancedFinancialAnalysisSystem()
    
    # If just showing tools, display tool info and exit
    if args.show_tools:
        tool_info = system.get_virtual_tool_info()
        print(f"Available Virtual Tools: {tool_info['count']}")
        for tool in tool_info['tools']:
            print(f"- {tool['name']}: {tool['description']} (confidence: {tool['confidence']:.2f})")
        print(f"\nVirtual Tool Stats: {json.dumps(tool_info['stats'], indent=2)}")
        return 0
    
    # Set the human input mode based on arguments
    human_input_mode = "TERMINATE" if args.interactive else "NEVER"
    
    print(f"Starting financial analysis on {args.data}")
    print(f"Task: {args.task}")
    
    try:
        # Run the analysis
        results = await system.run_analysis(
            data_source=args.data,
            task=args.task,
            sheet_name=args.sheet,
            human_input_mode=human_input_mode
        )
        
        # Save results if output path is provided
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Analysis results saved to {args.output}")
        
        print("Analysis completed successfully.")
        
        # Show virtual tool info
        tool_stats = results.get("virtual_tool_stats", {})
        print(f"\nVirtual Tool Stats:")
        print(f"- Total Tools: {tool_stats.get('total_tools', 0)}")
        print(f"- High Confidence Tools: {tool_stats.get('high_confidence_tools', 0)}")
        print(f"- New Potential Tool Patterns: {results.get('virtual_tool_candidates', 0)}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


def main():
    """
    Main entry point for the financial analysis system CLI.
    """
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())