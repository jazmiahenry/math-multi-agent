"""
Tool Registry for Financial Analysis Multi-Agent System

This module manages registration and discovery of tools for
OpenAI Assistants API integration, with support for TypeScript tools.
"""

from typing import Dict, Any, List, Callable, Optional
import asyncio
import json
import os
from python_agent.src.utils.logger import logger
from python_agent.src.tools.test_tools import call_math_tool  # Import existing tool caller

class ToolRegistry:
    """
    Manages registration and discovery of tools.
    
    This class provides a registry for tools that can be used by
    the OpenAI Assistants API via the MCP protocol.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools = {}
        self.tool_definitions = []
    
    def register_tool(self, 
                     name: str, 
                     description: str, 
                     parameters_schema: Dict[str, Any]) -> None:
        """
        Register a tool function.
        
        Args:
            name: Tool name
            description: Tool description
            parameters_schema: JSON Schema for tool parameters
        """
        # Create async wrapper for the tool function
        async def tool_wrapper(params: Dict[str, Any]) -> Dict[str, Any]:
            return call_math_tool(name, params)
        
        self.tools[name] = tool_wrapper
        
        # Create OpenAI-compatible tool definition
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters_schema
            }
        }
        
        self.tool_definitions.append(tool_def)
        logger.info(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """
        Get a tool function by name.
        
        Args:
            name: Tool name
            
        Returns:
            Optional[Callable]: Tool function or None if not found
        """
        return self.tools.get(name)
    
    def get_all_tools(self) -> Dict[str, Callable]:
        """
        Get all registered tool functions.
        
        Returns:
            Dict[str, Callable]: Map of tool names to functions
        """
        return self.tools
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool definitions.
        
        Returns:
            List[Dict[str, Any]]: Tool definitions for OpenAI API
        """
        return self.tool_definitions
    
    def register_math_tools(self) -> None:
        """Register all standard mathematical tools."""
        # Register mean tool
        self.register_tool(
            name="mean",
            description="Calculate the arithmetic mean of a list of numbers",
            parameters_schema={
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numbers to calculate the mean"
                    }
                },
                "required": ["numbers"]
            }
        )
        
        # Register median tool
        self.register_tool(
            name="median",
            description="Calculate the median of a list of numbers",
            parameters_schema={
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numbers to calculate the median"
                    }
                },
                "required": ["numbers"]
            }
        )
        
        # Register mode tool
        self.register_tool(
            name="mode",
            description="Find the most frequent values in a list of numbers",
            parameters_schema={
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numbers to find the mode"
                    }
                },
                "required": ["numbers"]
            }
        )
        
        # Register standard deviation tool
        self.register_tool(
            name="std_deviation",
            description="Calculate the standard deviation of a list of numbers",
            parameters_schema={
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numbers to calculate the standard deviation"
                    }
                },
                "required": ["numbers"]
            }
        )
        
        # Register probability tool
        self.register_tool(
            name="probability",
            description="Calculate probability distribution from frequencies",
            parameters_schema={
                "type": "object",
                "properties": {
                    "frequencies": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of frequency values"
                    }
                },
                "required": ["frequencies"]
            }
        )
        
        # Register eigenvalues/eigenvectors tool
        self.register_tool(
            name="eigen",
            description="Calculate eigenvalues and eigenvectors of a matrix",
            parameters_schema={
                "type": "object",
                "properties": {
                    "matrix": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"}
                        },
                        "description": "Square matrix in the form [[row1], [row2], ...]"
                    }
                },
                "required": ["matrix"]
            }
        )
        
        logger.info("Registered all standard mathematical tools")


# Create a global instance for convenience
registry = ToolRegistry()