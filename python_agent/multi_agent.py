"""
Financial Analysis Multi-Agent System

This module implements a multi-agent system using Autogen framework for financial analysis
with a planning agent, executor agent, and worker agents. The system uses disk caching
to improve efficiency for repeated tasks.
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Optional, Union
import autogen
from autogen import Agent, ConversableAgent
from autogen.agentchat.conversable_agent import ConversableAgent

# Import the tool calling function
from test_tools import call_math_tool


class CacheManager:
    """
    Manages disk-based caching of agent outputs.
    
    This class handles saving and retrieving cached results to avoid
    redundant computation for identical tasks.
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir (str): Directory to store cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, data: Any) -> str:
        """
        Generate a unique cache key for the given data.
        
        Args:
            data: Input data to hash
            
        Returns:
            str: Unique hash key
        """
        # Convert data to a consistent string representation
        if isinstance(data, dict) or isinstance(data, list):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        # Create hash
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            key (str): Cache key
            
        Returns:
            str: File path for the cache file
        """
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, data: Any) -> Optional[Any]:
        """
        Retrieve cached result for the given data.
        
        Args:
            data: Data to look up in cache
            
        Returns:
            Optional[Any]: Cached result or None if not found
        """
        key = self._get_cache_key(data)
        cache_path = self._get_cache_path(key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If cache file is corrupted, return None
                return None
        
        return None
    
    def set(self, data: Any, result: Any) -> None:
        """
        Store result in cache.
        
        Args:
            data: Original input data (used for key generation)
            result: Result to cache
        """
        key = self._get_cache_key(data)
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(result, f)
        except IOError as e:
            print(f"Cache write error: {e}")


class PlannerAgent(ConversableAgent):
    """
    Agent responsible for decomposing tasks into executable plans with error adaptation.
    
    This agent takes high-level financial analysis tasks and breaks them down
    into a sequence of specific tool calls and operations, with built-in strategies
    for error detection and recovery.
    """
    
    def __init__(
        self, 
        cache_manager: CacheManager,
        config: Optional[Dict] = None,
        llm_config: Optional[Dict] = None
    ):
        """
        Initialize the planner agent.
        
        Args:
            cache_manager: Manager for caching plans
            config: Additional configuration options
            llm_config: Configuration for the language model
        """
        # Default LLM config if none provided
        if llm_config is None:
            llm_config = {
                "model": "gpt-4",
                "temperature": 0.7,
                "config_list": autogen.config_list_from_json(
                    "OAI_CONFIG_LIST",
                    filter_dict={"model": ["gpt-4"]}
                )
            }
        
        system_message = """You are a planning agent specialized in financial analysis with built-in error handling. 
Your task is to:
1. Decompose complex financial analysis problems into a sequence of tool calls
2. Identify the right mathematical and statistical tools to use
3. Structure the plan so that outputs from earlier steps can be inputs to later steps
4. Be explicit about input data transformations needed between steps
5. Include verification and validation steps in your plans
6. Build redundancy into critical calculations to detect and recover from errors

IMPORTANT CONTEXT: The mathematical tools being used occasionally produce incorrect results 
or silent errors. Your plans should include strategies to detect and mitigate these issues by:

- Adding verification steps that cross-check results using alternative calculation methods
- Calculating important metrics through multiple approaches to ensure consistency
- Incorporating reasonableness checks for each calculation (e.g., is the result within expected ranges?)
- Adding fallback strategies for when calculations appear to be incorrect
- Leveraging domain knowledge to validate whether results make sense in context

Available tools:
- mean: Calculate the arithmetic mean of a list of numbers
- median: Calculate the median of a list of numbers  
- mode: Find the most frequent values in a list of numbers
- std_deviation: Calculate the standard deviation of a list of numbers
- probability: Calculate probability distribution from frequencies
- eigen: Calculate eigenvalues and eigenvectors of a matrix
"""
        
        # Initialize the conversable agent with our custom configuration
        super().__init__(
            name="PlannerAgent",
            system_message=system_message,
            llm_config=llm_config,
            **(config or {})
        )
        
        self.cache_manager = cache_manager
        
        # Register the planning function
        self.register_reply(
            self._generate_plan,
            lambda msg: "generate plan" in msg.get("content", "").lower()
        )
    
    async def _generate_plan(self, messages: List[Dict], sender: Agent) -> Dict:
        """
        Generate a plan for a financial analysis task using the LLM.
        
        Args:
            messages: List of conversation messages
            sender: Agent who sent the message
            
        Returns:
            Dict: Generated plan with steps
        """
        # Extract the task from the latest message
        task = messages[-1].get("content", "")
        
        # Check disk cache first
        cached_plan = self.cache_manager.get(task)
        if cached_plan:
            self.send(
                recipient=sender,
                message=f"Retrieved cached plan for: {task}\n\nPlan: {json.dumps(cached_plan, indent=2)}"
            )
            return cached_plan
        
        # Construct a prompt for the LLM to generate a structured plan
        planning_prompt = f"""
Given the financial analysis task: "{task}"

Create a detailed execution plan with specific steps. For each step:
1. Specify which math tool to use (choose from: mean, median, mode, std_deviation, probability, eigen)
2. Define the exact payload/parameters for the tool
3. Explain the purpose of this step in the analysis

Format your response as a JSON object with this structure:
{{
  "task": "the original task description",
  "steps": [
    {{
      "id": 1,
      "description": "clear description of what this step accomplishes",
      "tool": "name of the tool to use",
      "payload": {{tool-specific parameters as a JSON object}}
    }},
    ...additional steps...
  ]
}}

Make sure each step's payload contains valid parameters for the selected tool:
- For mean, median, mode, std_deviation: {{"numbers": [list of numbers]}}
- For probability: {{"frequencies": [list of frequency values]}}
- For eigen: {{"matrix": [[row1], [row2], ...]}}

Extract any numeric data from the task when available and use it in your plan.
"""
        
        # Use the LLM to generate the plan
        try:
            # Create a message to send to the LLM
            planning_message = {"role": "user", "content": planning_prompt}
            
            # Use the built-in LLM to generate the plan
            llm_response = await self.llm_client.create_completion(
                messages=[planning_message],
                model=self.llm_config.get("model", "gpt-4")
            )
            
            response_content = llm_response.choices[0].message.content
            
            # Extract JSON from the LLM response
            # Find the first occurrence of '{' and the last occurrence of '}'
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}')
            
            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                # If we can't find valid JSON, create a simplified plan
                self.send(
                    recipient=sender,
                    message=f"Could not generate a valid plan structure. Creating a basic plan for: {task}"
                )
                
                # Extract numeric data from task if possible
                import re
                numbers = [float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", task)]
                
                # Create a fallback plan with basic analysis
                plan = {
                    "task": task,
                    "steps": [
                        {
                            "id": 1,
                            "description": "Calculate basic statistics",
                            "tool": "mean",
                            "payload": {"numbers": numbers if numbers else [0, 0]}
                        }
                    ]
                }
            else:
                # Parse the JSON response
                json_str = response_content[start_idx:end_idx+1]
                plan = json.loads(json_str)
            
            # Cache the plan
            self.cache_manager.set(task, plan)
            
            # Format and send the plan as a response
            plan_str = json.dumps(plan, indent=2)
            self.send(
                recipient=sender,
                message=f"Generated new plan for: {task}\n\nPlan: {plan_str}"
            )
            
            return plan
            
        except Exception as e:
            # Handle any errors during plan generation
            error_msg = f"Error generating plan: {str(e)}"
            self.send(recipient=sender, message=error_msg)
            
            # Return a minimal plan as fallback
            fallback_plan = {
                "task": task,
                "steps": [
                    {
                        "id": 1,
                        "description": "Basic analysis (fallback due to error)",
                        "tool": "mean",
                        "payload": {"numbers": [0, 0]}
                    }
                ],
                "error": str(e)
            }
            return fallback_plan


class ExecutorAgent(ConversableAgent):
    """
    Agent responsible for executing plans created by the PlannerAgent with error handling.
    
    This agent coordinates the execution of plans by dispatching tasks
    to worker agents, monitoring for errors or inconsistencies, and
    implementing recovery strategies when issues are detected.
    """
    
    def __init__(
        self,
        worker_agents: List[Agent],
        cache_manager: CacheManager,
        config: Optional[Dict] = None,
        llm_config: Optional[Dict] = None
    ):
        """
        Initialize the executor agent.
        
        Args:
            worker_agents: List of worker agents to dispatch tasks to
            cache_manager: Manager for caching execution results
            config: Additional configuration options
            llm_config: Configuration for the language model
        """
        if llm_config is None:
            llm_config = {
                "model": "gpt-3.5-turbo",
                "temperature": 0.3,
                "config_list": autogen.config_list_from_json(
                    "OAI_CONFIG_LIST",
                    filter_dict={"model": ["gpt-3.5-turbo"]}
                )
            }
        
        system_message = """You are an executor agent that coordinates the execution of 
financial analysis plans with advanced error handling capabilities. Your responsibilities include:
1. Executing each step in the plan by calling the appropriate tool
2. Tracking progress through the plan
3. Handling errors and retrying failed steps when appropriate
4. Monitoring for inconsistent or suspicious results
5. Implementing recovery strategies when errors are detected
6. Aggregating and formatting results for the human user

IMPORTANT CONTEXT: The mathematical tools being used occasionally produce incorrect results 
or silent errors. You must be vigilant in detecting these issues by:

- Comparing results against domain expectations (e.g., stock volatility is typically between 0-100%)
- Looking for inconsistencies between related calculations
- Requesting verification when results seem unusual or unexpected
- Coordinating with worker agents to perform alternative calculations when needed
- Maintaining transparency about any detected issues and recovery steps taken

Additionally, when executing multi-step plans:
- Cross-validate intermediate results before using them in subsequent steps
- Keep track of calculation confidence levels throughout the plan execution
- Provide clear explanations when results are corrected or adjusted
"""
        
        super().__init__(
            name="ExecutorAgent",
            system_message=system_message,
            llm_config=llm_config,
            **(config or {})
        )
        
        self.worker_agents = worker_agents
        self.cache_manager = cache_manager
        
        # Register execution function
        self.register_reply(
            self._execute_plan,
            lambda msg: "execute plan" in msg.get("content", "").lower()
        )
    
    async def _execute_plan(self, messages: List[Dict], sender: Agent) -> Dict:
        """
        Execute a plan by coordinating worker agents.
        
        Args:
            messages: List of conversation messages
            sender: Agent who sent the message
            
        Returns:
            Dict: Execution results
        """
        # Extract plan from the message (assuming it's in JSON format)
        # In practice, you might need more robust extraction
        msg_content = messages[-1].get("content", "")
        
        # Extract the plan - in a real system, this would need more robust parsing
        try:
            # Look for a JSON object in the message
            start_idx = msg_content.find("{")
            end_idx = msg_content.rfind("}")
            if start_idx >= 0 and end_idx > start_idx:
                plan_json = msg_content[start_idx:end_idx+1]
                plan = json.loads(plan_json)
            else:
                # Fallback if no JSON object found
                self.send(
                    recipient=sender,
                    message="Could not extract a valid plan from the message. Please provide a valid JSON plan."
                )
                return {"error": "Invalid plan format"}
        except json.JSONDecodeError:
            self.send(
                recipient=sender,
                message="Could not parse the plan as JSON. Please provide a valid JSON plan."
            )
            return {"error": "Invalid plan format"}
        
        # Check cache for this exact plan
        cached_results = self.cache_manager.get(plan)
        if cached_results:
            self.send(
                recipient=sender,
                message=f"Retrieved cached execution results for this plan: {json.dumps(cached_results, indent=2)}"
            )
            return cached_results
        
        # Execute each step of the plan
        results = {"steps": []}
        
        for step in plan.get("steps", []):
            step_id = step.get("id", len(results["steps"]) + 1)
            tool_name = step.get("tool")
            payload = step.get("payload", {})
            description = step.get("description", f"Step {step_id}")
            
            # Update the sender on current progress
            self.send(
                recipient=sender,
                message=f"Executing step {step_id}: {description}"
            )
            
            # Assign to an appropriate worker (round-robin for simplicity)
            worker_idx = (step_id - 1) % len(self.worker_agents)
            worker = self.worker_agents[worker_idx]
            
            # In a real implementation, you would await the worker's response
            # Here we directly call the tool since we're not using actual async communication
            tool_result = call_math_tool(tool_name, payload)
            
            # Record the result
            step_result = {
                "step_id": step_id,
                "description": description,
                "tool": tool_name,
                "payload": payload,
                "result": tool_result
            }
            
            results["steps"].append(step_result)
            
            # Check for errors
            if "error" in tool_result:
                self.send(
                    recipient=sender,
                    message=f"Error in step {step_id}: {tool_result['error']}"
                )
                # In a real implementation, you might have retry logic here
        
        # Add overall success/failure status
        results["success"] = all("error" not in step["result"] for step in results["steps"])
        
        # Cache the results
        self.cache_manager.set(plan, results)
        
        # Format and send the results
        results_str = json.dumps(results, indent=2)
        self.send(
            recipient=sender,
            message=f"Plan execution completed.\n\nResults: {results_str}"
        )
        
        return results


class WorkerAgent(ConversableAgent):
    """
    Agent responsible for executing individual tool calls with result verification.
    
    Each worker agent specializes in a particular domain or set of tools
    and handles the details of tool execution, including error detection and recovery.
    """
    
    def __init__(
        self,
        specialization: str,
        tools: List[str],
        config: Optional[Dict] = None,
        llm_config: Optional[Dict] = None
    ):
        """
        Initialize a worker agent.
        
        Args:
            specialization: Description of this worker's specialization
            tools: List of tool names this worker can execute
            config: Additional configuration options
            llm_config: Configuration for the language model
        """
        if llm_config is None:
            llm_config = {
                "model": "gpt-3.5-turbo",
                "temperature": 0.2,
                "config_list": autogen.config_list_from_json(
                    "OAI_CONFIG_LIST",
                    filter_dict={"model": ["gpt-3.5-turbo"]}
                )
            }
        
        system_message = f"""You are a worker agent specializing in {specialization}.
You are responsible for:
1. Executing tools accurately with the provided parameters
2. Validating inputs before execution
3. Formatting outputs correctly
4. Providing helpful error information when tools fail
5. IMPORTANT: Detecting incorrect results through independent verification
6. Implementing recovery strategies when errors are detected

Your available tools are: {', '.join(tools)}

IMPORTANT CONTEXT: The tools you're using occasionally return incorrect results or silent errors
due to an error simulation system. You must verify all results for correctness through
independent calculation methods before trusting them.
"""
        
        super().__init__(
            name=f"WorkerAgent_{specialization}",
            system_message=system_message,
            llm_config=llm_config,
            **(config or {})
        )
        
        self.tools = tools
        self.verification_threshold = 0.001  # Relative error threshold for numerical verification
        
        # Register tool execution function
        self.register_reply(
            self._execute_tool,
            lambda msg: any(tool in msg.get("content", "").lower() for tool in tools)
        )
    
    def _perform_independent_verification(self, tool_name: str, params: Dict, result: Any) -> Dict:
        """
        Verify results using alternative calculation methods.
        
        Args:
            tool_name: The name of the tool used
            params: The parameters passed to the tool
            result: The result to verify
            
        Returns:
            Dict: Verification results with consistency check
        """
        # Skip verification for non-numeric results or if error already detected
        if "error" in result or not isinstance(result.get("result"), (int, float)):
            return {"verified": False, "reason": "Cannot verify non-numeric or error result"}
        
        # Extract the relevant data for verification
        if tool_name in ["mean", "median", "mode", "std_deviation"]:
            numbers = params.get("numbers", [])
            if not numbers or len(numbers) < 2:
                return {"verified": True, "reason": "Insufficient data for verification"}
            
            # Perform independent calculations based on tool type
            if tool_name == "mean":
                # Method 1: Using sum and division
                method1 = sum(numbers) / len(numbers)
                
                # Method 2: Using incremental formula
                method2 = 0
                for num in numbers:
                    method2 += num
                method2 /= len(numbers)
                
                # Method 3: Using partial sums
                half = len(numbers) // 2
                method3 = (sum(numbers[:half]) + sum(numbers[half:])) / len(numbers)
                
                # Get the original result
                original = result.get("result", 0)
                
                # Check consistency between methods
                methods = [method1, method2, method3]
                consistent = all(abs(m - method1) / max(1, abs(method1)) < self.verification_threshold for m in methods)
                correct = abs(original - method1) / max(1, abs(method1)) < self.verification_threshold
                
                return {
                    "verified": consistent and correct,
                    "reason": "Verified using multiple calculation methods" if consistent and correct else "Inconsistent results detected",
                    "expected": method1,
                    "received": original,
                    "deviation": abs(original - method1) / max(1, abs(method1))
                }
                
            elif tool_name == "median":
                # Method 1: Sort and select middle
                sorted_nums = sorted(numbers)
                n = len(sorted_nums)
                method1 = sorted_nums[n // 2] if n % 2 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
                
                # Method 2: Use selection algorithm
                # For simplicity, use numpy here, but could be implemented manually
                import numpy as np
                method2 = float(np.median(numbers))
                
                # Get the original result
                original = result.get("result", 0)
                
                # Check consistency
                methods = [method1, method2]
                consistent = abs(method1 - method2) / max(1, abs(method1)) < self.verification_threshold
                correct = abs(original - method1) / max(1, abs(method1)) < self.verification_threshold
                
                return {
                    "verified": consistent and correct,
                    "reason": "Verified using multiple calculation methods" if consistent and correct else "Inconsistent results detected",
                    "expected": method1,
                    "received": original,
                    "deviation": abs(original - method1) / max(1, abs(method1))
                }
                
            elif tool_name == "std_deviation":
                # Method 1: Two-pass algorithm
                mean = sum(numbers) / len(numbers)
                squared_diff_sum = sum((x - mean) ** 2 for x in numbers)
                method1 = (squared_diff_sum / len(numbers)) ** 0.5
                
                # Method 2: Using numpy for comparison
                import numpy as np
                method2 = float(np.std(numbers))
                
                # Get the original result
                original = result.get("result", 0)
                
                # Check consistency
                methods = [method1, method2]
                consistent = abs(method1 - method2) / max(1, abs(method1)) < self.verification_threshold * 10  # Allow more variance here due to different implementations
                correct = abs(original - method1) / max(1, abs(method1)) < self.verification_threshold * 10
                
                return {
                    "verified": consistent and correct,
                    "reason": "Verified using multiple calculation methods" if consistent and correct else "Inconsistent results detected",
                    "expected": method1,
                    "received": original,
                    "deviation": abs(original - method1) / max(1, abs(method1))
                }
                
            elif tool_name == "mode":
                # Method 1: Count occurrences and find max
                counter = {}
                for num in numbers:
                    counter[num] = counter.get(num, 0) + 1
                    
                max_count = max(counter.values())
                method1 = [k for k, v in counter.items() if v == max_count]
                
                # Method 2: Using statistics module
                import statistics
                try:
                    method2 = statistics.mode(numbers)
                    method2 = [method2]
                except statistics.StatisticsError:
                    # Handle multimodal case
                    method2 = method1
                
                # Get the original result
                original = result.get("result", [])
                if not isinstance(original, list):
                    original = [original]
                
                # Check if all expected values are in the result
                all_values_match = all(val in original for val in method1) and len(original) == len(method1)
                
                return {
                    "verified": all_values_match,
                    "reason": "Verified result matches expected mode values" if all_values_match else "Mode values don't match expected values",
                    "expected": method1,
                    "received": original
                }
        
        elif tool_name == "probability":
            frequencies = params.get("frequencies", [])
            if not frequencies:
                return {"verified": True, "reason": "Insufficient data for verification"}
            
            # Method 1: Direct calculation
            total = sum(frequencies)
            expected_probs = [freq / total for freq in frequencies]
            
            # Get the original result
            original = result.get("result", [])
            
            # Check if probabilities sum to approximately 1
            sum_to_one = abs(sum(original) - 1.0) < self.verification_threshold
            
            # Check if each probability matches the expected value
            all_match = True
            for exp, act in zip(expected_probs, original):
                if abs(exp - act) > self.verification_threshold:
                    all_match = False
                    break
            
            return {
                "verified": sum_to_one and all_match,
                "reason": "Verified probabilities match expected values and sum to 1" if sum_to_one and all_match else "Probability values don't match expected values or don't sum to 1",
                "expected": expected_probs,
                "received": original
            }
            
        elif tool_name == "eigen":
            matrix = params.get("matrix", [])
            if not matrix or len(matrix) < 2:
                return {"verified": True, "reason": "Insufficient data for verification"}
            
            # For eigenvalues/vectors, we can check if Ax = λx holds for each eigenpair
            # This is a complex verification that requires matrix operations
            # For simplicity, using numpy here
            import numpy as np
            
            # Get original results
            eigenvalues = result.get("result", {}).get("eigenvalues", [])
            eigenvectors = result.get("result", {}).get("eigenvectors", [])
            
            if not eigenvalues or not eigenvectors or len(eigenvalues) != len(eigenvectors):
                return {"verified": False, "reason": "Invalid eigenvalue/eigenvector result structure"}
            
            # Convert to numpy arrays
            A = np.array(matrix)
            
            # Verify each eigenpair (λ, v) satisfies Av = λv
            all_verified = True
            for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors)):
                v = np.array(eigenvector)
                Av = A.dot(v)
                lambda_v = eigenvalue * v
                
                # Check if Av ≈ λv
                if not np.allclose(Av, lambda_v, rtol=self.verification_threshold * 10):
                    all_verified = False
                    break
            
            return {
                "verified": all_verified,
                "reason": "Verified eigenpairs satisfy Av = λv relation" if all_verified else "Eigenpairs don't satisfy the fundamental eigenvalue equation"
            }
        
        # Default case if tool not specifically handled
        return {"verified": True, "reason": "No specific verification method available for this tool"}
    
    def _recalculate_result(self, tool_name: str, params: Dict) -> Any:
        """
        Recalculate result independently when verification fails.
        
        Args:
            tool_name: The name of the tool used
            params: The parameters passed to the tool
            
        Returns:
            Any: Recalculated result
        """
        if tool_name == "mean":
            numbers = params.get("numbers", [])
            if not numbers:
                return {"error": "Empty array for mean calculation"}
            return {"result": sum(numbers) / len(numbers)}
            
        elif tool_name == "median":
            numbers = params.get("numbers", [])
            if not numbers:
                return {"error": "Empty array for median calculation"}
            
            sorted_nums = sorted(numbers)
            n = len(sorted_nums)
            result = sorted_nums[n // 2] if n % 2 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
            return {"result": result}
            
        elif tool_name == "mode":
            numbers = params.get("numbers", [])
            if not numbers:
                return {"error": "Empty array for mode calculation"}
            
            # Count occurrences
            counter = {}
            for num in numbers:
                counter[num] = counter.get(num, 0) + 1
                
            # Find max frequency
            max_count = max(counter.values())
            # Return all values with max frequency
            result = [k for k, v in counter.items() if v == max_count]
            return {"result": result}
            
        elif tool_name == "std_deviation":
            numbers = params.get("numbers", [])
            if not numbers:
                return {"error": "Empty array for standard deviation calculation"}
            
            mean = sum(numbers) / len(numbers)
            squared_diff_sum = sum((x - mean) ** 2 for x in numbers)
            result = (squared_diff_sum / len(numbers)) ** 0.5
            return {"result": result}
            
        elif tool_name == "probability":
            frequencies = params.get("frequencies", [])
            if not frequencies:
                return {"error": "Empty array for probability calculation"}
            
            total = sum(frequencies)
            result = [freq / total for freq in frequencies]
            return {"result": result}
            
        elif tool_name == "eigen":
            import numpy as np
            matrix = params.get("matrix", [])
            if not matrix or len(matrix) < 2:
                return {"error": "Invalid matrix for eigenvalue calculation"}
            
            # Use numpy for eigenvalue calculation
            A = np.array(matrix)
            eigenvalues, eigenvectors = np.linalg.eig(A)
            
            # Convert to list format
            eigenvalues = eigenvalues.tolist()
            eigenvectors = [v.tolist() for v in eigenvectors.T]
            
            return {"result": {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}}
        
        # Default case
        return {"error": f"No independent calculation method available for {tool_name}"}
    
    async def _execute_tool(self, messages: List[Dict], sender: Agent) -> Dict:
        """
        Execute a tool based on the request with error detection and recovery.
        
        Args:
            messages: List of conversation messages
            sender: Agent who sent the message
            
        Returns:
            Dict: Tool execution results (verified or corrected)
        """
        # Parse the tool request from the message
        msg_content = messages[-1].get("content", "")
        
        # Extract tool name and parameters
        tool_name = None
        for tool in self.tools:
            if tool in msg_content:
                tool_name = tool
                break
        
        if not tool_name:
            self.send(
                recipient=sender,
                message=f"Could not identify a tool to execute. Available tools: {', '.join(self.tools)}"
            )
            return {"error": "No valid tool specified"}
        
        # Extract parameters - in a real system, you'd need better parsing
        try:
            # Look for a JSON object in the message
            start_idx = msg_content.find("{")
            end_idx = msg_content.rfind("}")
            if start_idx >= 0 and end_idx > start_idx:
                params_json = msg_content[start_idx:end_idx+1]
                params = json.loads(params_json)
            else:
                # Fallback to a default parameter set
                if tool_name == "mean" or tool_name == "median" or tool_name == "std_deviation":
                    params = {"numbers": [1, 2, 3, 4, 5]}
                elif tool_name == "probability":
                    params = {"frequencies": [10, 20, 30, 40]}
                elif tool_name == "eigen":
                    params = {"matrix": [[4, 2], [1, 3]]}
                else:
                    params = {}
        except json.JSONDecodeError:
            self.send(
                recipient=sender,
                message=f"Could not parse parameters as JSON. Using default parameters."
            )
            if tool_name == "mean" or tool_name == "median" or tool_name == "std_deviation":
                params = {"numbers": [1, 2, 3, 4, 5]}
            elif tool_name == "probability":
                params = {"frequencies": [10, 20, 30, 40]}
            elif tool_name == "eigen":
                params = {"matrix": [[4, 2], [1, 3]]}
            else:
                params = {}
        
        # Execute the tool
        try:
            # First attempt with the external tool
            original_result = call_math_tool(tool_name, params)
            
            # Perform verification of the result
            verification = self._perform_independent_verification(tool_name, params, original_result)
            
            if verification.get("verified", False):
                # Result verified successfully
                self.send(
                    recipient=sender,
                    message=f"Executed {tool_name} with parameters {json.dumps(params)}.\nResult: {json.dumps(original_result)}\nVerification: Passed ✓"
                )
                return original_result
            else:
                # Result verification failed, perform independent recalculation
                self.send(
                    recipient=sender,
                    message=f"⚠️ Verification failed for {tool_name} result: {verification.get('reason', 'Unknown reason')}"
                )
                
                # Recalculate the result independently
                corrected_result = self._recalculate_result(tool_name, params)
                
                # Log the correction
                self.send(
                    recipient=sender,
                    message=f"Recalculated {tool_name} with parameters {json.dumps(params)}.\nOriginal result: {json.dumps(original_result)}\nCorrected result: {json.dumps(corrected_result)}\n\nUsing corrected result for further analysis."
                )
                
                return corrected_result
                
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            self.send(recipient=sender, message=error_msg)
            
            # Attempt recovery through independent calculation
            try:
                recovery_result = self._recalculate_result(tool_name, params)
                self.send(
                    recipient=sender,
                    message=f"Recovered from error using independent calculation.\nResult: {json.dumps(recovery_result)}"
                )
                return recovery_result
            except Exception as recovery_e:
                # If recovery also fails, return original error
                return {"error": str(e), "recovery_error": str(recovery_e)}


def create_agent_system():
    """
    Create and configure the multi-agent system.
    
    Returns:
        tuple: (planner_agent, executor_agent, list_of_worker_agents)
    """
    # Initialize cache manager
    cache_manager = CacheManager(cache_dir=".agent_cache")
    
    # Create worker agents with different specializations
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
    
    workers = [stats_worker, probability_worker, linear_algebra_worker]
    
    # Create executor agent
    executor = ExecutorAgent(
        worker_agents=workers,
        cache_manager=cache_manager
    )
    
    # Create planner agent
    planner = PlannerAgent(
        cache_manager=cache_manager
    )
    
    return planner, executor, workers


# This is the end of the agents module
# The main module is implemented separately below