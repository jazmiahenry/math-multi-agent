"""
Event Handler for OpenAI Assistants API

This module implements a custom event handler for the OpenAI Assistants API
that tracks tool calls and their results.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from openai import OpenAI, AssistantEventHandler
from openai.types.beta.threads.runs import ToolCall

from python_agent.src.rl.dpo_trainer import DPOTrainer
from python_agent.src.utils.logger import logger

class EventHandler(AssistantEventHandler):
    """
    Handle events from OpenAI Assistant API.
    
    This class provides custom handling for Assistant API events,
    particularly tool calls.
    """
    
    def __init__(self, 
                client: OpenAI,
                tool_registry: Dict[str, callable],
                dpo_trainer: Optional[DPOTrainer] = None):
        """
        Initialize event handler.
        
        Args:
            client: OpenAI client
            tool_registry: Registry of available tools
            dpo_trainer: Optional DPO trainer
        """
        super().__init__()
        self.client = client
        self.tool_registry = tool_registry
        self.dpo_trainer = dpo_trainer
        self.tool_calls = []
        self.tool_outputs = []
    
    async def on_tool_call(self, tool_call: ToolCall) -> None:
        """
        Handle a tool call from the assistant.
    
        Args:
            tool_call: Tool call object
        """
        # Record the tool call
        self.tool_calls.append(tool_call)
    
        # Process the tool call
        output = await self.process_tool_call(tool_call)
    
        # Submit the tool output
        await self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread_id,
            run_id=self.run_id,
            tool_outputs=[
                {
                    "tool_call_id": tool_call.id,
                    "output": output
                }
            ]
        )
    
    def get_tool_sequence(self) -> List[Dict]:
        """
        Get the sequence of tool calls and outputs.
        
        Returns:
            List[Dict]: Sequence of tool call steps
        """
        sequence = []
        
        for i, (call, output) in enumerate(zip(self.tool_calls, self.tool_outputs)):
            step = {
                "step_id": i + 1,
                "tool": call.function.name,
                "payload": json.loads(call.function.arguments) if call.function.arguments else {},
                "result": output["result"]
            }
            sequence.append(step)
        
        return sequence
    
    async def on_exception(self, exception: Exception) -> None:
        """
        Handle exceptions during the run.
        
        Args:
            exception: The exception that occurred
        """
        logger.error(f"Exception during assistant run: {str(exception)}")
    
    async def on_end(self) -> None:
        """
        Handle the end of the run.
        """
        logger.info(f"Run completed with {len(self.tool_calls)} tool calls")
        
        # Additional metrics or logging could be added here
        if self.tool_calls:
            success_rate = sum(1 for o in self.tool_outputs if "error" not in o["result"]) / len(self.tool_outputs)
            logger.info(f"Tool call success rate: {success_rate:.2f}")

    async def process_tool_call(self, tool_call: ToolCall) -> str:
        function_name = tool_call.function.name
        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            function_args = {}
            logger.error(f"Error parsing arguments for tool call: {tool_call.function.arguments}")

        logger.info(f"Tool call: {function_name} with args: {function_args}")

        # Normalize input for different tool types
        if function_name in ['mean', 'median', 'mode', 'std_deviation']:
            function_args = function_args.get('numbers', [])
        elif function_name == 'probability':
            function_args = function_args.get('frequencies', [])
        elif function_name == 'eigen':
            function_args = function_args.get('matrix', [])

        # Find the appropriate tool
        tool_fn = self.tool_registry.get(function_name)

        if tool_fn is None:
            result = {"error": f"Unknown tool: {function_name}"}
        else:
            # Execute the tool
            try:
                result = await tool_fn(function_args)
            except Exception as e:
                logger.error(f"Error executing tool {function_name}: {str(e)}")
                result = {"error": str(e)}

        # Record the output for tracking
        self.tool_outputs.append({
            "tool_call_id": tool_call.id,
            "function_name": function_name,
            "args": function_args,
            "result": result
        })

        return json.dumps(result)