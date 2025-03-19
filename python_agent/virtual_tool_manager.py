"""
Virtual Tool Manager for Financial Analysis Multi-Agent System

This module provides functionality to create, store, and execute virtual tools
that represent successful sequences of tool calls for solving specific types
of mathematical problems.
"""

import os
import json
import hashlib
import inspect
import logging
import functools
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("virtual_tools.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VirtualToolManager")


class VirtualTool:
    """
    Represents a learned sequence of tool calls bundled as a single operation.
    
    A VirtualTool encapsulates a sequence of mathematical operations that has been
    proven to reliably solve a specific class of problems. It includes metadata
    about the problem pattern it solves and performance metrics.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        tool_sequence: List[Dict],
        problem_pattern: Dict,
        confidence: float = 0.0,
        execution_count: int = 0,
        success_rate: float = 0.0,
        created_at: Optional[str] = None
    ):
        """
        Initialize a virtual tool.
        
        Args:
            name: Unique name for the virtual tool
            description: Human-readable description of what the tool does
            tool_sequence: List of tool calls with their parameters and transformations
            problem_pattern: Pattern that defines what problems this tool can solve
            confidence: Confidence level in this tool's reliability (0.0-1.0)
            execution_count: Number of times this tool has been executed
            success_rate: Success rate of this tool (0.0-1.0)
            created_at: Timestamp when this tool was created
        """
        self.name = name
        self.description = description
        self.tool_sequence = tool_sequence
        self.problem_pattern = problem_pattern
        self.confidence = confidence
        self.execution_count = execution_count
        self.success_rate = success_rate
        self.created_at = created_at or datetime.now().isoformat()
        
        # Generate a unique signature for this tool
        self.signature = self._generate_signature()
    
    def _generate_signature(self) -> str:
        """
        Generate a unique signature for this virtual tool based on its sequence.
        
        Returns:
            str: Hash signature representing this tool
        """
        # Create a deterministic representation of the tool sequence
        sequence_str = json.dumps(self.tool_sequence, sort_keys=True)
        pattern_str = json.dumps(self.problem_pattern, sort_keys=True)
        
        # Generate a hash
        signature = hashlib.md5(f"{sequence_str}|{pattern_str}".encode('utf-8')).hexdigest()
        return signature
    
    def update_metrics(self, success: bool) -> None:
        """
        Update the usage metrics for this tool.
        
        Args:
            success: Whether the execution was successful
        """
        self.execution_count += 1
        # Update success rate using a moving average approach
        if self.execution_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            # Weight recent executions more heavily
            self.success_rate = (0.9 * self.success_rate) + (0.1 * (1.0 if success else 0.0))
        
        # Update confidence based on execution count and success rate
        # Higher execution count and success rate = higher confidence
        self.confidence = min(0.99, (1.0 - (1.0 / (1.0 + self.execution_count))) * self.success_rate)
    
    def to_dict(self) -> Dict:
        """
        Convert this virtual tool to a dictionary for serialization.
        
        Returns:
            Dict: Dictionary representation of this tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "tool_sequence": self.tool_sequence,
            "problem_pattern": self.problem_pattern,
            "confidence": self.confidence,
            "execution_count": self.execution_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at,
            "signature": self.signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VirtualTool':
        """
        Create a virtual tool from a dictionary.
        
        Args:
            data: Dictionary representation of a virtual tool
            
        Returns:
            VirtualTool: Reconstructed VirtualTool instance
        """
        return cls(
            name=data["name"],
            description=data["description"],
            tool_sequence=data["tool_sequence"],
            problem_pattern=data["problem_pattern"],
            confidence=data["confidence"],
            execution_count=data["execution_count"],
            success_rate=data["success_rate"],
            created_at=data["created_at"]
        )


class VirtualToolManager:
    """
    Manages the creation, storage, retrieval, and execution of virtual tools.
    
    This class provides the interface for the multi-agent system to learn from
    successful tool sequences and reuse them when similar problems are encountered.
    """
    
    def __init__(self, storage_path: str = ".virtual_tools"):
        """
        Initialize the virtual tool manager.
        
        Args:
            storage_path: Directory to store virtual tools
        """
        self.storage_path = storage_path
        self.virtual_tools: Dict[str, VirtualTool] = {}
        self.tool_registry: Dict[str, Callable] = {}
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing virtual tools
        self._load_virtual_tools()
    
    def register_tool(self, name: str, tool_fn: Callable) -> None:
        """
        Register a basic tool function that can be used in virtual tools.
        
        Args:
            name: Name of the tool
            tool_fn: Function that implements the tool
        """
        self.tool_registry[name] = tool_fn
        logger.info(f"Registered base tool: {name}")
    
    def _load_virtual_tools(self) -> None:
        """Load all virtual tools from the storage directory."""
        try:
            tool_files = [f for f in os.listdir(self.storage_path) if f.endswith('.json')]
            for file_name in tool_files:
                file_path = os.path.join(self.storage_path, file_name)
                with open(file_path, 'r') as f:
                    try:
                        tool_data = json.load(f)
                        virtual_tool = VirtualTool.from_dict(tool_data)
                        self.virtual_tools[virtual_tool.signature] = virtual_tool
                        logger.info(f"Loaded virtual tool: {virtual_tool.name} (confidence: {virtual_tool.confidence:.2f})")
                    except Exception as e:
                        logger.error(f"Error loading virtual tool from {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading virtual tools: {str(e)}")
    
    def _save_virtual_tool(self, virtual_tool: VirtualTool) -> None:
        """
        Save a virtual tool to storage.
        
        Args:
            virtual_tool: The virtual tool to save
        """
        try:
            file_path = os.path.join(self.storage_path, f"{virtual_tool.signature}.json")
            with open(file_path, 'w') as f:
                json.dump(virtual_tool.to_dict(), f, indent=2)
            logger.info(f"Saved virtual tool: {virtual_tool.name} ({virtual_tool.signature})")
        except Exception as e:
            logger.error(f"Error saving virtual tool {virtual_tool.name}: {str(e)}")
    
    def create_virtual_tool(
        self,
        name: str,
        description: str,
        tool_sequence: List[Dict],
        problem_pattern: Dict,
        reasoning: str
    ) -> VirtualTool:
        """
        Create a new virtual tool from a successful sequence of tool calls.
        
        Args:
            name: Name for the new virtual tool
            description: Description of what the tool does
            tool_sequence: Sequence of tool calls with parameters
            problem_pattern: Pattern that defines what problems this tool can solve
            reasoning: Explanation of why this sequence works for this problem type
            
        Returns:
            VirtualTool: The newly created virtual tool
        """
        # Create the virtual tool
        virtual_tool = VirtualTool(
            name=name,
            description=description,
            tool_sequence=tool_sequence,
            problem_pattern=problem_pattern,
            confidence=0.5,  # Initial confidence - will be updated with usage
            execution_count=1,
            success_rate=1.0  # Assume first creation is successful
        )
        
        # Log the creation with reasoning
        logger.info(f"Created new virtual tool: {name}")
        logger.info(f"Tool sequence: {json.dumps(tool_sequence, indent=2)}")
        logger.info(f"Problem pattern: {json.dumps(problem_pattern, indent=2)}")
        logger.info(f"Reasoning: {reasoning}")
        
        # Store the virtual tool
        self.virtual_tools[virtual_tool.signature] = virtual_tool
        self._save_virtual_tool(virtual_tool)
        
        return virtual_tool
    
    def find_matching_tool(self, problem_description: str, params: Dict) -> Optional[VirtualTool]:
        """
        Find a virtual tool that matches a given problem.
        
        Args:
            problem_description: Description of the problem to solve
            params: Parameters of the problem
            
        Returns:
            Optional[VirtualTool]: Matching virtual tool or None if no match found
        """
        matching_tools = []
        
        for tool in self.virtual_tools.values():
            # Calculate pattern match score
            match_score = self._calculate_match_score(tool.problem_pattern, problem_description, params)
            
            # If match score is above threshold and tool has sufficient confidence
            if match_score >= 0.7 and tool.confidence >= 0.6:
                matching_tools.append((tool, match_score))
        
        # Sort by match score and confidence
        matching_tools.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)
        
        if matching_tools:
            best_match, score = matching_tools[0]
            logger.info(f"Found matching virtual tool: {best_match.name} (match score: {score:.2f}, confidence: {best_match.confidence:.2f})")
            return best_match
        
        return None
    
    def _calculate_match_score(self, pattern: Dict, problem_description: str, params: Dict) -> float:
        """
        Calculate how well a problem matches a virtual tool's pattern.
        
        Args:
            pattern: The pattern from a virtual tool
            problem_description: Description of the current problem
            params: Parameters for the current problem
            
        Returns:
            float: Match score between 0.0 and 1.0
        """
        # This is a simplified matching algorithm
        # In a production system, this would use more sophisticated NLP and pattern matching
        
        # Check for keyword matches in the problem description
        keywords = pattern.get("keywords", [])
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in problem_description.lower())
        keyword_score = keyword_matches / max(1, len(keywords)) if keywords else 0.0
        
        # Check for parameter structure matches
        required_params = pattern.get("required_params", [])
        param_matches = sum(1 for param in required_params if param in params)
        param_score = param_matches / max(1, len(required_params)) if required_params else 0.0
        
        # Check for data type matches
        data_types = pattern.get("data_types", {})
        type_matches = 0
        type_total = len(data_types)
        
        if type_total > 0:
            for param_name, expected_type in data_types.items():
                if param_name in params:
                    param_value = params[param_name]
                    
                    # Check if the parameter value matches the expected type
                    if expected_type == "list" and isinstance(param_value, list):
                        type_matches += 1
                    elif expected_type == "number" and isinstance(param_value, (int, float)):
                        type_matches += 1
                    elif expected_type == "string" and isinstance(param_value, str):
                        type_matches += 1
                    elif expected_type == "matrix" and isinstance(param_value, list) and all(isinstance(row, list) for row in param_value):
                        type_matches += 1
        
        type_score = type_matches / max(1, type_total) if type_total > 0 else 0.0
        
        # Overall match score with weighted components
        overall_score = (0.4 * keyword_score) + (0.3 * param_score) + (0.3 * type_score)
        
        return overall_score
    
    async def execute_virtual_tool(
        self, 
        virtual_tool: VirtualTool, 
        params: Dict,
        execute_tool_fn: Callable
    ) -> Dict:
        """
        Execute a virtual tool with the given parameters.
        
        Args:
            virtual_tool: The virtual tool to execute
            params: Parameters for tool execution
            execute_tool_fn: Function to execute individual tools
            
        Returns:
            Dict: Results of the virtual tool execution
        """
        logger.info(f"Executing virtual tool: {virtual_tool.name}")
        
        # Initialize results storage for the sequence
        step_results = []
        final_result = None
        success = True
        
        try:
            # Execute each step in the tool sequence
            for i, step in enumerate(virtual_tool.tool_sequence):
                step_number = i + 1
                tool_name = step.get("tool")
                
                # Get parameter mapping for this step
                param_mapping = step.get("param_mapping", {})
                
                # Prepare parameters for this step
                step_params = {}
                
                for param_name, param_source in param_mapping.items():
                    if param_source.get("type") == "input":
                        # Parameter comes from the input params
                        input_name = param_source.get("name")
                        if input_name in params:
                            step_params[param_name] = params[input_name]
                    elif param_source.get("type") == "previous_step":
                        # Parameter comes from a previous step result
                        prev_step = param_source.get("step")
                        result_path = param_source.get("result_path", [])
                        
                        if prev_step <= len(step_results):
                            prev_result = step_results[prev_step - 1]
                            
                            # Navigate the result path to extract the specific value
                            value = prev_result
                            for path_item in result_path:
                                if isinstance(value, dict) and path_item in value:
                                    value = value[path_item]
                                else:
                                    break
                            
                            step_params[param_name] = value
                
                # Apply any transformations defined for this step
                transformations = step.get("transformations", [])
                for transform in transformations:
                    transform_type = transform.get("type")
                    param_name = transform.get("param")
                    
                    if param_name in step_params:
                        if transform_type == "sort":
                            step_params[param_name] = sorted(step_params[param_name])
                        elif transform_type == "reverse":
                            step_params[param_name] = list(reversed(step_params[param_name]))
                        elif transform_type == "filter_positive":
                            step_params[param_name] = [x for x in step_params[param_name] if x > 0]
                        # Add more transformations as needed
                
                # Execute the tool for this step
                step_result = await execute_tool_fn(tool_name, step_params)
                step_results.append(step_result)
                
                # Check for errors
                if "error" in step_result:
                    logger.warning(f"Error in step {step_number} of virtual tool {virtual_tool.name}: {step_result['error']}")
                    success = False
                    break
            
            # If all steps completed successfully, the final result is from the last step
            if success and step_results:
                final_result = step_results[-1]
                
                # Apply any final transformations
                final_transform = virtual_tool.tool_sequence[-1].get("final_transform")
                if final_transform:
                    # Apply final transformation logic here
                    pass
        
        except Exception as e:
            logger.error(f"Error executing virtual tool {virtual_tool.name}: {str(e)}")
            success = False
            final_result = {"error": f"Virtual tool execution failed: {str(e)}"}
        
        # Update the tool's metrics
        virtual_tool.update_metrics(success)
        self._save_virtual_tool(virtual_tool)
        
        # Return the final result
        return final_result or {"error": "Virtual tool execution failed"}
    
    def get_all_virtual_tools(self) -> List[VirtualTool]:
        """
        Get all registered virtual tools.
        
        Returns:
            List[VirtualTool]: List of all virtual tools
        """
        return list(self.virtual_tools.values())
    
    def get_virtual_tool_stats(self) -> Dict:
        """
        Get statistics about virtual tools.
        
        Returns:
            Dict: Statistics about virtual tools
        """
        total_tools = len(self.virtual_tools)
        high_confidence_tools = sum(1 for tool in self.virtual_tools.values() if tool.confidence >= 0.8)
        total_executions = sum(tool.execution_count for tool in self.virtual_tools.values())
        
        return {
            "total_tools": total_tools,
            "high_confidence_tools": high_confidence_tools,
            "total_executions": total_executions,
            "tools_by_confidence": {
                "low": sum(1 for t in self.virtual_tools.values() if t.confidence < 0.5),
                "medium": sum(1 for t in self.virtual_tools.values() if 0.5 <= t.confidence < 0.8),
                "high": high_confidence_tools
            }
        }