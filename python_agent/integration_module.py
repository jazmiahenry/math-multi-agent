"""
Integration Module for Financial Analysis Multi-Agent System

This module integrates all components of the enhanced multi-agent system,
including the error detection and recovery system and the virtual tool learning system.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import autogen

# Import components
from virtual_tool_manager import VirtualToolManager
from learning_agent import LearningAgent
from test_tools import call_math_tool


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("financial_agents.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FinancialAgents")


class EnhancedFinancialSystem:
    """
    Enhanced financial analysis system with error detection and virtual tool learning.
    
    This class integrates all components of the multi-agent system including:
    - Basic agent framework (planner, executor, workers)
    - Error detection and recovery system
    - Virtual tool learning and reuse system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the enhanced financial system.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize virtual tool manager
        self.virtual_tool_manager = VirtualToolManager(
            storage_path=self.config.get("virtual_tool_storage", ".virtual_tools")
        )
        
        # Register base tools with the virtual tool manager
        self._register_base_tools()
        
        # Initialize cache manager
        from autogen_implementation import CacheManager
        self.cache_manager = CacheManager(
            cache_dir=self.config.get("cache_dir", ".agent_cache")
        )
        
        # Create agents
        self._create_agents()
    
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
    
    def _register_base_tools(self) -> None:
        """Register base mathematical tools with the virtual tool manager."""
        # Create wrappers around the call_math_tool function for each tool
        def mean_tool(params):
            return call_math_tool("mean", params)
        
        def median_tool(params):
            return call_math_tool("median", params)
        
        def mode_tool(params):
            return call_math_tool("mode", params)
        
        def std_deviation_tool(params):
            return call_math_tool("std_deviation", params)
        
        def probability_tool(params):
            return call_math_tool("probability", params)
        
        def eigen_tool(params):
            return call_math_tool("eigen", params)
        
        # Register tools with the virtual tool manager
        self.virtual_tool_manager.register_tool("mean", mean_tool)
        self.virtual_tool_manager.register_tool("median", median_tool)
        self.virtual_tool_manager.register_tool("mode", mode_tool)
        self.virtual_tool_manager.register_tool("std_deviation", std_deviation_tool)
        self.virtual_tool_manager.register_tool("probability", probability_tool)
        self.virtual_tool_manager.register_tool("eigen", eigen_tool)
    
    def _create_agents(self) -> None:
        """Create all agents for the system."""
        from autogen_implementation import PlannerAgent, ExecutorAgent, WorkerAgent
        
        # Create worker agents with error detection capabilities
        stats_worker = WorkerAgent(
            specialization="statistical analysis",
            tools=["mean", "median", "mode", "std_deviation"]
        )
        
        probability_worker = WorkerAgent(
            specialization="probability theory",
            tools=["probability"]
        )
        
        linear_algebra_worker = WorkerAgent(
            specialization="linear algebra",
            tools=["eigen"]
        )
        
        self.workers = [stats_worker, probability_worker, linear_algebra_worker]
        
        # Create executor agent with error recovery functionality
        self.executor = ExecutorAgent(
            worker_agents=self.workers,
            cache_manager=self.cache_manager
        )
        
        # Create planner agent with error adaptation strategies
        self.planner = PlannerAgent(
            cache_manager=self.cache_manager
        )
        
        # Create learning agent for virtual tool creation
        self.learning_agent = LearningAgent(
            virtual_tool_manager=self.virtual_tool_manager
        )
    
    async def execute_tool_with_learning(
        self,
        tool_name: str,
        params: Dict,
        task_description: str
    ) -> Dict:
        """
        Execute a tool and use the execution for learning.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            task_description: Description of the task being performed
            
        Returns:
            Dict: Result of the tool execution
        """
        # First check if there's a matching virtual tool
        virtual_tool = self.virtual_tool_manager.find_matching_tool(
            task_description,
            params
        )
        
        if virtual_tool:
            logger.info(f"Using virtual tool {virtual_tool.name} for task: {task_description}")
            
            # Execute the virtual tool
            result = await self.virtual_tool_manager.execute_virtual_tool(
                virtual_tool,
                params,
                self._execute_individual_tool
            )
            
            return result
        
        # If no virtual tool matches, execute the regular tool
        start_time = datetime.now()
        result = await self._execute_individual_tool(tool_name, params)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Record this execution for learning
        self.learning_agent.observe_execution(
            task=task_description,
            steps=[{"tool": tool_name, "params": params}],
            results=[result],
            success="error" not in result
        )
        
        return result
    
    async def _execute_individual_tool(self, tool_name: str, params: Dict) -> Dict:
        """
        Execute an individual tool with error detection.
        
        This function delegates to the appropriate worker agent based on the tool.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            
        Returns:
            Dict: Result of the tool execution
        """
        # Find a worker that can handle this tool
        for worker in self.workers:
            if tool_name in worker.tools:
                # Create a message to the worker
                message = {
                    "content": f"Execute tool {tool_name} with parameters {json.dumps(params)}"
                }
                
                # Use the worker's tool execution method
                result = await worker._execute_tool([message], self.executor)
                return result
        
        # If no worker can handle this tool, call it directly
        # This is a fallback and shouldn't normally be needed
        try:
            return call_math_tool(tool_name, params)
        except Exception as e:
            return {"error": f"Error executing {tool_name}: {str(e)}"}
    
    async def execute_plan_with_learning(
        self,
        plan: Dict,
        task_description: str
    ) -> Dict:
        """
        Execute a plan and use the execution for learning.
        
        Args:
            plan: Plan to execute with steps and parameters
            task_description: Description of the task being performed
            
        Returns:
            Dict: Results of the plan execution
        """
        steps = plan.get("steps", [])
        results = {"steps": []}
        
        # Check if there's a matching virtual tool for the entire plan
        virtual_tool = self.virtual_tool_manager.find_matching_tool(
            task_description,
            {"plan": plan}  # Pass the whole plan as a parameter
        )
        
        if virtual_tool:
            logger.info(f"Using virtual tool {virtual_tool.name} for task: {task_description}")
            
            # Extract parameters from the plan
            plan_params = {}
            for step in steps:
                if "payload" in step:
                    for param_name, param_value in step["payload"].items():
                        plan_params[param_name] = param_value
            
            # Execute the virtual tool
            result = await self.virtual_tool_manager.execute_virtual_tool(
                virtual_tool,
                plan_params,
                self._execute_individual_tool
            )
            
            return {"steps": [{"result": result}], "success": "error" not in result}
        
        # Execute each step in the plan
        step_results = []
        success = True
        
        for i, step in enumerate(steps):
            step_id = step.get("id", i + 1)
            tool_name = step.get("tool")
            params = step.get("payload", {})
            description = step.get("description", f"Step {step_id}")
            
            logger.info(f"Executing step {step_id}: {description}")
            
            # Execute the tool
            result = await self._execute_individual_tool(tool_name, params)
            
            # Record the result
            step_result = {
                "step_id": step_id,
                "description": description,
                "tool": tool_name,
                "params": params,
                "result": result
            }
            
            step_results.append(step_result)
            results["steps"].append(step_result)
            
            # Check for errors
            if "error" in result:
                logger.warning(f"Error in step {step_id}: {result['error']}")
                success = False
                break
        
        # Add overall success/failure status
        results["success"] = success
        
        # Record this execution for learning
        self.learning_agent.observe_execution(
            task=task_description,
            steps=steps,
            results=step_results,
            success=success
        )
        
        return results
    
    async def analyze_for_virtual_tools(self) -> List[Dict]:
        """
        Analyze execution history to identify potential virtual tools.
        
        Returns:
            List[Dict]: List of identified patterns that could become virtual tools
        """
        # Delegate to the learning agent
        patterns = await self.learning_agent.analyze_observations()
        
        # Log identified patterns
        if patterns:
            logger.info(f"Identified {len(patterns)} potential virtual tool patterns")
            for i, pattern in enumerate(patterns):
                logger.info(f"Pattern {i+1}: {pattern['sequence']} (used {pattern['count']} times)")
        
        return patterns
    
    def get_virtual_tool_stats(self) -> Dict:
        """
        Get statistics about virtual tools.
        
        Returns:
            Dict: Statistics about virtual tools
        """
        return self.virtual_tool_manager.get_virtual_tool_stats()


# Create an execution function that delegates to a specific agent
async def execute_math_analysis(task: str, data: List[float]) -> Dict:
    """
    Execute a mathematical analysis task using the enhanced agent system.
    
    Args:
        task: Description of the analysis task
        data: Numerical data to analyze
        
    Returns:
        Dict: Results of the analysis
    """
    # Initialize the enhanced system
    system = EnhancedFinancialSystem()
    
    # Create a user proxy agent for interaction
    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",  # Disable human input for automation
        max_consecutive_auto_reply=10,
        system_message="You are a financial analyst who needs help analyzing data."
    )
    
    # Set up a group chat including all agents
    groupchat = autogen.GroupChat(
        agents=[user_proxy, system.planner, system.executor, system.learning_agent] + system.workers,
        messages=[],
        max_round=50
    )
    
    manager = autogen.GroupChatManager(groupchat=groupchat)
    
    # Construct the task message with data
    task_message = f"""
I need to analyze the following financial data: {data}
Task: {task}

Generate a comprehensive analysis plan using the appropriate mathematical tools.
"""
    
    # Initiate the conversation to generate and execute the plan
    conversation_result = await user_proxy.initiate_chat(
        manager,
        message=task_message
    )
    
    # The plan is generated by the planner agent and executed by the executor agent
    # After execution, we check if we can create a virtual tool
    patterns = await system.analyze_for_virtual_tools()
    
    if patterns:
        # We found a potentially reusable pattern
        pattern = patterns[0]
        logger.info(f"Identified reusable pattern: {pattern['sequence']}")
        
        # In a real system, we would interact with the learning agent to create a virtual tool
        logger.info(f"Could create virtual tool from pattern: {pattern['potential_name']}")
    
    # Extract results from the conversation
    # In a real implementation, you would parse the conversation to extract structured results
    results = {
        "task": task,
        "data": data,
        "conversation": conversation_result,
        "virtual_tool_candidates": len(patterns) if patterns else 0
    }
    
    return results


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def run_example():
        task = "Calculate the volatility and central tendency of stock prices"
        data = [10, 15, 20, 18, 25, 30, 28]
        
        results = await execute_math_analysis(task, data)
        print(json.dumps(results, indent=2))
    
    asyncio.run(run_example())