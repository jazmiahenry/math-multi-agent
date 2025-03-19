"""
Learning Agent for Financial Analysis Multi-Agent System

This module implements a learning agent that observes tool call sequences,
identifies successful patterns, and creates virtual tools for future reuse.
"""

import json
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
import autogen
from autogen import Agent, ConversableAgent

from virtual_tool_manager import VirtualToolManager, VirtualTool


class LearningAgent(ConversableAgent):
    """
    Agent responsible for learning from successful tool call sequences.
    
    This agent observes the interactions between other agents, identifies patterns
    in successful tool call sequences, and creates virtual tools that can be reused
    for similar problems in the future.
    """
    
    def __init__(
        self,
        virtual_tool_manager: VirtualToolManager,
        config: Optional[Dict] = None,
        llm_config: Optional[Dict] = None
    ):
        """
        Initialize the learning agent.
        
        Args:
            virtual_tool_manager: Manager for virtual tools
            config: Additional configuration options
            llm_config: Configuration for the language model
        """
        if llm_config is None:
            llm_config = {
                "model": "gpt-4",
                "temperature": 0.2,
                "config_list": autogen.config_list_from_json(
                    "OAI_CONFIG_LIST",
                    filter_dict={"model": ["gpt-4"]}
                )
            }
        
        system_message = """You are a learning agent that identifies patterns in successful 
mathematical tool sequences and creates reusable virtual tools. Your responsibilities include:

1. Observing tool call sequences performed by other agents
2. Identifying patterns in successful problem-solving approaches
3. Extracting the essential structure of these tool sequences
4. Creating virtual tools that can be reused for similar problems
5. Providing clear documentation and explanations for the virtual tools you create

Your goal is to continually improve the system's efficiency by learning from experience.
When you identify a successful pattern, describe:
- What mathematical problem this pattern solves
- The essential steps and their purpose
- How parameters flow between steps
- What makes this pattern generalizable
- How to recognize similar problems in the future
"""
        
        super().__init__(
            name="LearningAgent",
            system_message=system_message,
            llm_config=llm_config,
            **(config or {})
        )
        
        self.virtual_tool_manager = virtual_tool_manager
        self.observation_buffer = []
        self.known_patterns = set()
        
        # Register message handlers
        self.register_reply(
            self._handle_observation,
            lambda msg: "observe_sequence" in msg.get("content", "").lower()
        )
        
        self.register_reply(
            self._handle_create_virtual_tool,
            lambda msg: "create_virtual_tool" in msg.get("content", "").lower()
        )
    
    async def _handle_observation(self, messages: List[Dict], sender: Agent) -> str:
        """
        Handle observations of tool call sequences.
        
        Args:
            messages: List of conversation messages
            sender: Agent who sent the message
            
        Returns:
            str: Response to the observation
        """
        # Extract the latest message
        msg_content = messages[-1].get("content", "")
        
        # Try to extract the sequence information from the message
        try:
            # Look for JSON object in the message
            start_idx = msg_content.find("{")
            end_idx = msg_content.rfind("}")
            
            if start_idx >= 0 and end_idx > start_idx:
                observation_json = msg_content[start_idx:end_idx+1]
                observation = json.loads(observation_json)
                
                # Add to observation buffer
                self.observation_buffer.append(observation)
                
                # Check if this forms a pattern worth learning
                pattern_assessment = await self._assess_pattern_value(observation)
                
                if pattern_assessment["should_learn"]:
                    # Generate a response about the pattern
                    response = f"""
I've observed a potentially valuable pattern in the tool sequence for solving {pattern_assessment['problem_type']}:

Pattern Value Assessment:
- Reusability: {pattern_assessment['reusability']}/10
- Efficiency: {pattern_assessment['efficiency']}/10
- Reliability: {pattern_assessment['reliability']}/10
- Distinctiveness: {pattern_assessment['distinctiveness']}/10

{pattern_assessment['reasoning']}

This pattern has been added to my learning buffer. I can create a virtual tool from this pattern if you'd like.
To create a virtual tool, respond with:
"create_virtual_tool: {pattern_assessment['suggested_name']}"
"""
                    return response
                else:
                    return f"""
I've observed the tool sequence, but it doesn't appear to form a distinctive reusable pattern:

{pattern_assessment['reasoning']}

I'll continue to observe and learn from more complex sequences.
"""
            else:
                return "I couldn't find valid JSON data in your observation. Please ensure the tool sequence is properly formatted."
        
        except Exception as e:
            return f"Error processing observation: {str(e)}"
    
    async def _assess_pattern_value(self, observation: Dict) -> Dict:
        """
        Assess whether a tool sequence forms a valuable pattern worth learning.
        
        Args:
            observation: Observation of a tool sequence
            
        Returns:
            Dict: Assessment results
        """
        # This would normally use the LLM to evaluate the pattern's value
        # For now, use some simple heuristics
        
        # Extract key components from the observation
        task = observation.get("task", "")
        steps = observation.get("steps", [])
        success = observation.get("success", False)
        
        # Calculate a signature for this pattern
        tools_used = [step.get("tool") for step in steps]
        tool_signature = "→".join(tools_used)
        
        # Check if we already know this pattern
        if tool_signature in self.known_patterns:
            return {
                "should_learn": False,
                "reasoning": "This is a pattern we've already observed and assessed.",
                "problem_type": "unknown"
            }
        
        # Only consider successful sequences
        if not success:
            return {
                "should_learn": False,
                "reasoning": "This sequence was not successful, so we won't learn from it.",
                "problem_type": "unknown"
            }
        
        # Simple heuristics for pattern value:
        # 1. Sequences with multiple steps are more interesting
        # 2. Sequences that transform data between steps are more valuable
        # 3. Tasks mentioning specific mathematical concepts are more identifiable
        
        # Count steps
        step_count = len(steps)
        
        # Check for data transformations between steps
        has_transformations = False
        for step in steps:
            if "transformations" in step or "param_mapping" in step:
                has_transformations = True
                break
        
        # Look for mathematical concepts in the task
        math_concepts = [
            "mean", "average", "median", "mode", "standard deviation", "variance",
            "correlation", "regression", "probability", "distribution", "eigenvalue",
            "vector", "matrix", "volatility", "trend", "forecast", "prediction"
        ]
        
        identified_concepts = [concept for concept in math_concepts if concept.lower() in task.lower()]
        
        # Determine problem type based on concepts and tools
        problem_type = "unknown mathematical problem"
        suggested_name = "UnknownMathTool"
        
        if "volatility" in identified_concepts or "std_deviation" in tools_used:
            problem_type = "volatility calculation"
            suggested_name = "VolatilityAnalyzer"
        elif any(concept in identified_concepts for concept in ["mean", "average", "median"]) or any(tool in tools_used for tool in ["mean", "median"]):
            problem_type = "central tendency analysis"
            suggested_name = "CentralTendencyAnalyzer"
        elif "probability" in identified_concepts or "probability" in tools_used:
            problem_type = "probability analysis"
            suggested_name = "ProbabilityCalculator"
        elif "eigen" in tools_used or any(concept in identified_concepts for concept in ["eigenvalue", "vector", "matrix"]):
            problem_type = "matrix analysis"
            suggested_name = "MatrixAnalyzer"
        
        # Calculate scores for different aspects of pattern value
        reusability = min(10, step_count * 2 + len(identified_concepts) * 2)
        efficiency = 10 if step_count >= 3 else step_count * 3
        reliability = 8  # Assume reliability is good since it was successful
        distinctiveness = len(identified_concepts) * 2 + (5 if has_transformations else 0)
        
        # Decision on whether to learn this pattern
        should_learn = (
            success and
            step_count >= 2 and
            (has_transformations or len(identified_concepts) >= 1) and
            reusability + efficiency + reliability + distinctiveness >= 25
        )
        
        # Add to known patterns if assessed
        if should_learn:
            self.known_patterns.add(tool_signature)
        
        return {
            "should_learn": should_learn,
            "reusability": reusability,
            "efficiency": efficiency,
            "reliability": reliability,
            "distinctiveness": distinctiveness,
            "problem_type": problem_type,
            "suggested_name": suggested_name,
            "reasoning": f"""
This is a {step_count}-step sequence for {problem_type} that {'has' if has_transformations else 'does not have'} data transformations between steps.
It uses the tools: {', '.join(tools_used)}.
The task mentions these mathematical concepts: {', '.join(identified_concepts) if identified_concepts else 'none specifically'}.
"""
        }
    
    def _extract_required_params(self, steps: List[Dict]) -> List[str]:
        """
        Extract required parameters from a sequence of tool steps.
        
        Args:
            steps: List of tool steps
            
        Returns:
            List[str]: List of required parameter names
        """
        required_params = set()
        
        for step in steps:
            if "param_mapping" in step:
                for param_mapping in step["param_mapping"].values():
                    if param_mapping.get("type") == "input":
                        input_name = param_mapping.get("name")
                        if input_name:
                            required_params.add(input_name)
        
        return list(required_params)
    
    def _determine_data_types(self, steps: List[Dict]) -> Dict[str, str]:
        """
        Determine data types for parameters in a sequence of tool steps.
        
        Args:
            steps: List of tool steps
            
        Returns:
            Dict[str, str]: Mapping of parameter names to data types
        """
        data_types = {}
        
        for step in steps:
            tool_name = step.get("tool")
            
            # Infer data types based on the tool
            if tool_name in ["mean", "median", "mode", "std_deviation"]:
                data_types["numbers"] = "list"
            elif tool_name == "probability":
                data_types["frequencies"] = "list"
            elif tool_name == "eigen":
                data_types["matrix"] = "matrix"
        
        return data_types
    
    async def _handle_create_virtual_tool(self, messages: List[Dict], sender: Agent) -> str:
        """
        Handle requests to create a virtual tool from an observed pattern.
        
        Args:
            messages: List of conversation messages
            sender: Agent who sent the message
            
        Returns:
            str: Response about the created virtual tool
        """
        # Extract the latest message
        msg_content = messages[-1].get("content", "")
        
        # Extract the suggested name using regex
        match = re.search(r"create_virtual_tool:\s*(\w+)", msg_content)
        if not match:
            return "I couldn't find a valid tool name in your request. Please use the format: create_virtual_tool: ToolName"
        
        suggested_name = match.group(1)
        
        # Ensure we have observations to work with
        if not self.observation_buffer:
            return "I don't have any observed tool sequences to create a virtual tool from. Please provide observations first."
        
        # Get the most recent observation
        observation = self.observation_buffer[-1]
        
        # Use LLM to generate a proper description and problem pattern
        # In a real implementation, this would use the LLM
        # For this implementation, we'll create a simplified version
        
        # Extract information
        task = observation.get("task", "")
        steps = observation.get("steps", [])
        success = observation.get("success", False)
        
        if not success:
            return "I can't create a virtual tool from an unsuccessful tool sequence. Please provide a successful sequence."
        
        # Create a tool name with proper format
        tool_name = suggested_name
        if not tool_name.endswith("Tool"):
            tool_name += "Tool"
        
        # Generate description
        description = f"Executes a sequence of {len(steps)} tools to solve {task}"
        
        # Extract keywords from the task
        keywords = []
        math_terms = [
            "mean", "average", "median", "mode", "standard deviation", "variance",
            "correlation", "regression", "probability", "distribution", "eigenvalue",
            "vector", "matrix", "volatility", "trend", "forecast", "prediction"
        ]
        
        for term in math_terms:
            if term.lower() in task.lower():
                keywords.append(term)
        
        # Create a problem pattern
        problem_pattern = {
            "keywords": keywords,
            "required_params": self._extract_required_params(steps),
            "data_types": self._determine_data_types(steps)
        }
        
        # Generate reasoning about why this sequence works
        tools_used = [step.get("tool") for step in steps]
        tool_sequence = " → ".join(tools_used)
        
        # Generate reasoning for the virtual tool
        reasoning = f"""
This virtual tool combines {len(steps)} mathematical operations in sequence to solve problems related to {', '.join(keywords) if keywords else 'financial analysis'}.

The tool sequence ({tool_sequence}) is efficient because:
1. It follows a logical progression from raw data to meaningful insights
2. Each step builds upon the previous step's outputs
3. The sequence handles the core computational needs for this type of problem
4. It incorporates proper validation and error handling

This sequence can be generalized to similar problems that require the same analytical approach.
For future problems with keywords like {', '.join(keywords) if keywords else 'these mathematical concepts'}, 
this virtual tool can be applied directly, saving time and reducing the chance of errors.
"""
        
        # Create the virtual tool
        virtual_tool = self.virtual_tool_manager.create_virtual_tool(
            name=tool_name,
            description=description,
            tool_sequence=steps,
            problem_pattern=problem_pattern,
            reasoning=reasoning
        )
        
        # Return information about the created tool
        return f"""
✅ Successfully created a new virtual tool: **{virtual_tool.name}**

**Description**: {virtual_tool.description}

**Problem Pattern**:
- Keywords: {', '.join(problem_pattern['keywords']) if problem_pattern['keywords'] else 'None specified'}
- Required Parameters: {', '.join(problem_pattern['required_params']) if problem_pattern['required_params'] else 'None specified'}
- Data Types: {json.dumps(problem_pattern['data_types'])}

**Tool Sequence**:
{tool_sequence}

**Reasoning**:
{reasoning}

This virtual tool has been stored and can now be used when similar problems are encountered.
Current confidence level: {virtual_tool.confidence:.2f}
"""
    
    def observe_execution(
        self, 
        task: str, 
        steps: List[Dict], 
        results: List[Dict], 
        success: bool
    ) -> None:
        """
        Observe the execution of a sequence of tools.
        
        Args:
            task: Description of the task being performed
            steps: List of tool steps that were executed
            results: List of results from each step
            success: Whether the sequence was successful
        """
        # Create an observation record
        observation = {
            "task": task,
            "steps": steps,
            "results": results,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to observation buffer
        self.observation_buffer.append(observation)
        
        # Limit buffer size
        if len(self.observation_buffer) > 100:
            self.observation_buffer = self.observation_buffer[-100:]
    
    async def analyze_observations(self) -> List[Dict]:
        """
        Analyze collected observations to identify patterns.
        
        Returns:
            List[Dict]: List of identified patterns
        """
        # This would typically use the LLM to analyze patterns
        # For now, return a simple summary
        
        patterns = []
        tool_sequences = {}
        
        # Group observations by tool sequence
        for obs in self.observation_buffer:
            if obs.get("success", False):
                tools_used = [step.get("tool") for step in obs.get("steps", [])]
                tool_sig = "→".join(tools_used)
                
                if tool_sig not in tool_sequences:
                    tool_sequences[tool_sig] = []
                
                tool_sequences[tool_sig].append(obs)
        
        # Find sequences that occur multiple times
        for sig, observations in tool_sequences.items():
            if len(observations) >= 2:
                # This sequence has been used successfully multiple times
                sample_obs = observations[0]
                
                patterns.append({
                    "sequence": sig,
                    "count": len(observations),
                    "sample_task": sample_obs.get("task", ""),
                    "steps": sample_obs.get("steps", []),
                    "potential_name": f"Pattern_{hashlib.md5(sig.encode()).hexdigest()[:8]}"
                })
        
        return patterns