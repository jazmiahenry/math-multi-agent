"""
AgentOps Integration for Financial Multi-Agent System

This module provides integration with AgentOps for monitoring, debugging, and evaluating
the performance of the financial multi-agent system.
"""

import os
import json
import uuid
from typing import Dict, List, Any, Optional
import agentops
from agentops.schemas.span import SpanKind

# Add this to requirements.txt:
# agentops>=0.2.0


class AgentOpsMonitor:
    """
    Provides monitoring and evaluation of the multi-agent system using AgentOps.
    
    This class handles instrumentation of agent conversations, tool executions,
    and performance metrics to enable evaluation and debugging of the system.
    """
    
    def __init__(
        self, 
        project_name: str = "financial-multi-agent",
        api_key: Optional[str] = None,
        enable_tracing: bool = True
    ):
        """
        Initialize the AgentOps monitor.
        
        Args:
            project_name: Name of the project in AgentOps
            api_key: AgentOps API key (defaults to AGENTOPS_API_KEY env var)
            enable_tracing: Whether to enable detailed tracing
        """
        self.project_name = project_name
        self.api_key = api_key or os.getenv("AGENTOPS_API_KEY")
        self.enable_tracing = enable_tracing
        self.session_id = str(uuid.uuid4())
        
        # Initialize AgentOps
        agentops.init(
            api_key=self.api_key,
            project=self.project_name
        )
        
        # Create a new session
        self.trace = agentops.start_trace(
            session_id=self.session_id,
            metadata={
                "system": "financial-multi-agent",
                "version": "1.0.0"
            }
        )
        
        # Active spans by agent name/tool
        self.active_spans = {}
    
    def start_task(self, task: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking a new task.
        
        Args:
            task: Description of the task
            metadata: Additional task metadata
            
        Returns:
            str: Task ID
        """
        task_id = str(uuid.uuid4())
        
        agentops.start_span(
            name=f"Task: {task}",
            span_id=task_id,
            kind=SpanKind.TASK,
            metadata=metadata or {}
        )
        
        return task_id
    
    def end_task(self, task_id: str, success: bool, result: Optional[Dict] = None) -> None:
        """
        End tracking for a task.
        
        Args:
            task_id: ID of the task to end
            success: Whether the task was successful
            result: Task result data
        """
        agentops.end_span(
            span_id=task_id,
            metadata={
                "success": success,
                "result": json.dumps(result) if result else None
            }
        )
    
    def start_agent_action(
        self, 
        agent_name: str, 
        action: str, 
        inputs: Optional[Dict] = None
    ) -> str:
        """
        Start tracking an agent action.
        
        Args:
            agent_name: Name of the agent
            action: Type of action (e.g., "planning", "execution")
            inputs: Input data for the action
            
        Returns:
            str: Span ID for this action
        """
        span_id = f"{agent_name}_{action}_{uuid.uuid4()}"
        
        agentops.start_span(
            name=f"{agent_name}: {action}",
            span_id=span_id,
            kind=SpanKind.AGENT,
            metadata={
                "agent_name": agent_name,
                "action_type": action,
                "inputs": json.dumps(inputs) if inputs else None
            }
        )
        
        self.active_spans[f"{agent_name}_{action}"] = span_id
        return span_id
    
    def end_agent_action(
        self, 
        agent_name: str, 
        action: str, 
        success: bool, 
        outputs: Optional[Dict] = None,
        metrics: Optional[Dict] = None
    ) -> None:
        """
        End tracking for an agent action.
        
        Args:
            agent_name: Name of the agent
            action: Type of action
            success: Whether the action was successful
            outputs: Output data from the action
            metrics: Performance metrics
        """
        span_key = f"{agent_name}_{action}"
        if span_key in self.active_spans:
            span_id = self.active_spans[span_key]
            
            metadata = {
                "success": success,
                "outputs": json.dumps(outputs) if outputs else None
            }
            
            # Add any metrics
            if metrics:
                for key, value in metrics.items():
                    metadata[key] = value
            
            agentops.end_span(
                span_id=span_id,
                metadata=metadata
            )
            
            # Remove from active spans
            del self.active_spans[span_key]
    
    def track_tool_execution(
        self, 
        tool_name: str, 
        params: Dict, 
        result: Dict, 
        execution_time: float,
        error_detected: bool = False,
        correction_applied: bool = False
    ) -> None:
        """
        Track a tool execution.
        
        Args:
            tool_name: Name of the tool
            params: Parameters for the tool
            result: Result of the tool execution
            execution_time: Time taken to execute the tool (seconds)
            error_detected: Whether an error was detected
            correction_applied: Whether a correction was applied
        """
        span_id = f"tool_{tool_name}_{uuid.uuid4()}"
        
        # Start and immediately end a span for the tool execution
        agentops.start_span(
            name=f"Tool: {tool_name}",
            span_id=span_id,
            kind=SpanKind.TOOL
        )
        
        agentops.end_span(
            span_id=span_id,
            metadata={
                "tool_name": tool_name,
                "parameters": json.dumps(params),
                "result": json.dumps(result),
                "execution_time_seconds": execution_time,
                "error_detected": error_detected,
                "correction_applied": correction_applied
            }
        )
    
    def track_virtual_tool_creation(
        self,
        virtual_tool_name: str,
        tool_sequence: List[Dict],
        problem_pattern: Dict,
        confidence: float,
        reasoning: str
    ) -> None:
        """
        Track the creation of a new virtual tool.
        
        Args:
            virtual_tool_name: Name of the virtual tool
            tool_sequence: Sequence of tools in the virtual tool
            problem_pattern: Pattern that defines what problems this tool can solve
            confidence: Initial confidence in the tool
            reasoning: Reasoning for creating this tool
        """
        span_id = f"vtool_creation_{uuid.uuid4()}"
        
        # Start and immediately end a span for the virtual tool creation
        agentops.start_span(
            name=f"VirtualTool Creation: {virtual_tool_name}",
            span_id=span_id,
            kind=SpanKind.AGENT
        )
        
        agentops.end_span(
            span_id=span_id,
            metadata={
                "virtual_tool_name": virtual_tool_name,
                "tool_sequence": json.dumps(tool_sequence),
                "problem_pattern": json.dumps(problem_pattern),
                "confidence": confidence,
                "reasoning": reasoning,
                "event_type": "virtual_tool_creation"
            }
        )
    
    def track_virtual_tool_execution(
        self,
        virtual_tool_name: str,
        problem: str,
        params: Dict,
        result: Dict,
        execution_time: float,
        success: bool
    ) -> None:
        """
        Track the execution of a virtual tool.
        
        Args:
            virtual_tool_name: Name of the virtual tool
            problem: Problem being solved
            params: Parameters for the virtual tool
            result: Result of the virtual tool execution
            execution_time: Time taken to execute the virtual tool (seconds)
            success: Whether the execution was successful
        """
        span_id = f"vtool_execution_{uuid.uuid4()}"
        
        # Start and immediately end a span for the virtual tool execution
        agentops.start_span(
            name=f"VirtualTool Execution: {virtual_tool_name}",
            span_id=span_id,
            kind=SpanKind.TOOL
        )
        
        agentops.end_span(
            span_id=span_id,
            metadata={
                "virtual_tool_name": virtual_tool_name,
                "problem": problem,
                "parameters": json.dumps(params),
                "result": json.dumps(result),
                "execution_time_seconds": execution_time,
                "success": success,
                "event_type": "virtual_tool_execution"
            }
        )
    
    def track_error_detection(
        self,
        tool_name: str,
        params: Dict,
        original_result: Dict,
        corrected_result: Dict,
        verification_method: str,
        deviation: float
    ) -> None:
        """
        Track the detection and correction of an error.
        
        Args:
            tool_name: Name of the tool with the error
            params: Parameters for the tool
            original_result: Original erroneous result
            corrected_result: Corrected result
            verification_method: Method used to verify and correct the result
            deviation: Measure of deviation between original and corrected results
        """
        span_id = f"error_detection_{uuid.uuid4()}"
        
        # Start and immediately end a span for the error detection
        agentops.start_span(
            name=f"Error Detection: {tool_name}",
            span_id=span_id,
            kind=SpanKind.TASK
        )
        
        agentops.end_span(
            span_id=span_id,
            metadata={
                "tool_name": tool_name,
                "parameters": json.dumps(params),
                "original_result": json.dumps(original_result),
                "corrected_result": json.dumps(corrected_result),
                "verification_method": verification_method,
                "deviation": deviation,
                "event_type": "error_detection"
            }
        )
    
    def add_metric(self, name: str, value: Any) -> None:
        """
        Add a custom metric to the current trace.
        
        Args:
            name: Name of the metric
            value: Value of the metric
        """
        agentops.update_trace(
            metadata={name: value}
        )
    
    def end_session(self, success: bool, summary: Optional[Dict] = None) -> None:
        """
        End the AgentOps monitoring session.
        
        Args:
            success: Whether the overall session was successful
            summary: Summary of the session results
        """
        # Close any remaining active spans
        for span_key, span_id in list(self.active_spans.items()):
            agentops.end_span(
                span_id=span_id,
                metadata={"warning": "Span closed automatically at session end"}
            )
            del self.active_spans[span_key]
        
        # Update the trace with final metadata
        agentops.update_trace(
            metadata={
                "overall_success": success,
                "summary": json.dumps(summary) if summary else None
            }
        )
        
        # End the trace
        agentops.end_trace()


# Example usage of AgentOps monitoring in the multi-agent system
def instrument_enhanced_system(system, monitor):
    """
    Instrument an EnhancedFinancialSystem with AgentOps monitoring.
    
    Args:
        system: The enhanced financial system to instrument
        monitor: The AgentOps monitor
    """
    # Wrap execute_individual_tool to track tool executions
    original_execute_tool = system._execute_individual_tool
    
    async def monitored_execute_tool(tool_name, params):
        start_time = __import__('time').time()
        span_id = monitor.start_agent_action(f"WorkerAgent_{tool_name}", "execute", params)
        
        try:
            result = await original_execute_tool(tool_name, params)
            execution_time = __import__('time').time() - start_time
            
            # Check if error was detected
            error_detected = False
            correction_applied = False
            
            # For worker agents, check for verification info in the logs
            # This is a simplified approach - in practice, you'd need to capture
            # this information during the actual verification process
            for worker in system.workers:
                if tool_name in worker.tools:
                    # Check if the worker's last action was verification
                    # This is placeholder logic - implement based on your actual system
                    pass
            
            monitor.track_tool_execution(
                tool_name=tool_name,
                params=params,
                result=result,
                execution_time=execution_time,
                error_detected=error_detected,
                correction_applied=correction_applied
            )
            
            monitor.end_agent_action(
                agent_name=f"WorkerAgent_{tool_name}",
                action="execute",
                success="error" not in result,
                outputs=result,
                metrics={"execution_time": execution_time}
            )
            
            return result
        except Exception as e:
            execution_time = __import__('time').time() - start_time
            monitor.end_agent_action(
                agent_name=f"WorkerAgent_{tool_name}",
                action="execute",
                success=False,
                outputs={"error": str(e)},
                metrics={"execution_time": execution_time}
            )
            raise
    
    # Replace the method with our monitored version
    system._execute_individual_tool = monitored_execute_tool
    
    # Similarly wrap other methods to track planning, execution, and learning
    # This would be implemented based on your specific system architecture
    
    return system