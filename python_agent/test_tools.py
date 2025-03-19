import requests
import json

def call_math_tool(tool_name, data):
    """
    Call a math tool from the TypeScript service.
    
    Args:
        tool_name (str): Name of the tool to call ('mean', 'median', etc.)
        data (dict): Data to send in the request
        
    Returns:
        dict: Response from the service or error information
    """
    try:
        response = requests.post(
            f"http://localhost:3000/tool/{tool_name}", 
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Return the response data
        return response.json()
    
    except requests.exceptions.HTTPError as e:
        # Handle server errors (like the simulated errors)
        try:
            error_data = response.json()
            print(f"Tool error: {error_data.get('error', 'Unknown error')}")
            return {"error": error_data.get('error', 'Unknown error')}
        except:
            print(f"HTTP Error: {e}")
            return {"error": str(e)}
    
    except requests.exceptions.RequestException as e:
        # Handle connection errors
        print(f"Request Error: {e}")
        return {"error": str(e)}

# Example usage:
if __name__ == "__main__":
    # Calculate mean
    mean_result = call_math_tool("mean", {"numbers": [5, 10, 15, 20, 25]})
    if "error" not in mean_result:
        print(f"Mean: {mean_result['result']}")
    
    # Calculate median
    median_result = call_math_tool("median", {"numbers": [5, 10, 15, 20, 25]})
    if "error" not in median_result:
        print(f"Median: {median_result['result']}")
    
    # Calculate mode
    mode_result = call_math_tool("mode", {"numbers": [1, 2, 2, 3, 3, 3, 4, 5]})
    if "error" not in mode_result:
        print(f"Mode: {mode_result['result']}")
    
    # Calculate standard deviation
    std_dev_result = call_math_tool("std_deviation", {"numbers": [5, 10, 15, 20, 25]})
    if "error" not in std_dev_result:
        print(f"Standard Deviation: {std_dev_result['result']}")
    
    # Calculate probability distribution
    prob_result = call_math_tool("probability", {"frequencies": [10, 20, 30, 40]})
    if "error" not in prob_result:
        print(f"Probability Distribution: {prob_result['result']}")
    
    # Calculate eigenvalues and eigenvectors
    eigen_result = call_math_tool("eigen", {"matrix": [[4, 2], [1, 3]]})
    if "error" not in eigen_result:
        print(f"Eigenvalues: {eigen_result['result']['eigenvalues']}")
        print(f"Eigenvectors: {eigen_result['result']['eigenvectors']}")