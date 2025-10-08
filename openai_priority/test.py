from openai import OpenAI
from openai_priority.testdata import prompts
from dotenv import load_dotenv

import os
import time
import pandas as pd

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Prepare results storage
results = []

def measure_latency(service_type="standard"):
    latencies = []
    for prompt in prompts:
        start_time = time.time()
        
        if service_type == "standard":
            response = client.responses.create(
                model="gpt-4o",
                input=prompt,
                max_output_tokens=200
            )
        elif service_type == "priority":
            response = client.responses.create(
                model="gpt-4o",
                input=prompt,
                max_output_tokens=200,
                service_tier="priority"
            )
        else:
            raise ValueError("service_type must be 'standard' or 'priority'")
        
        latency = time.time() - start_time
        latencies.append(latency)
    
    return latencies

if __name__ == "__main__":
    print("Measuring Standard Service Latency...")
    standard_latencies = measure_latency("standard")
    
    print("Measuring Priority Service Latency...")
    priority_latencies = measure_latency("priority")
    
    # Save results to CSV
    df = pd.DataFrame({
        "prompt": prompts,
        "standard_latency": standard_latencies,
        "priority_latency": priority_latencies
    })
    
    df.to_csv("latency_comparison.csv", index=False)
    print("Latency results saved to latency_comparison.csv")
