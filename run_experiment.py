import mlflow
import subprocess
import json

def execute_experiment(model_name, temperature, ticker_count, run_name):
    print(f"🚀 Starting Experiment: {run_name}")
    
    # 1. Initialize MLflow Run
    mlflow.set_experiment("Magnus_Financial_Swarm")
    
    with mlflow.start_run(run_name=run_name):
        
        # 2. Log Hyperparameters (The "Ingredients")
        mlflow.log_param("model", model_name)
        mlflow.log_param("temperature", temperature)
        mlflow.log_param("ticker_count", ticker_count)
        mlflow.log_param("rag_enabled", True)
        
        # 3. Execute the Swarm Pipeline
        print("\n⚙️ Executing Autonomous Swarm...")
        # Note: In a production environment, you would pass model_name and temp 
        # as arguments to run_swarm.py via argparse. 
        subprocess.run(["python", "run_swarm.py"], check=True)
        
        # 4. Execute the Ground Truth Evaluator
        print("\n📊 Executing Ground Truth Evaluation...")
        # Capture the terminal output from ground_truth.py
        result = subprocess.run(
            ["python", "ground_truth.py"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # 5. Extract the Final Accuracy from the terminal output using basic string parsing
        accuracy = 0.0
        for line in result.stdout.split('\n'):
            print(line) # Print the table to terminal so you still see it
            if "Final Swarm Predictive Accuracy" in line:
                # Extract the float value from "Accuracy: 60.00%"
                accuracy_str = line.split(":")[-1].replace("%", "").strip()
                accuracy = float(accuracy_str)
                break
                
        # 6. Log Metrics & Artifacts (The "Results")
        mlflow.log_metric("accuracy_percentage", accuracy)
        
        # Save the actual debate JSON as an artifact so you can read exactly what the AI 
        # said during this specific run months from now.
        mlflow.log_artifact("master_debate_results.json")
        mlflow.log_artifact("evaluation_data.csv")
        
        print(f"\n✅ Experiment Logged Successfully. Accuracy: {accuracy}%")

if __name__ == "__main__":
    # Define the parameters for this specific run
    # You will change these variables each time you test a new configuration
    MODEL = "llama-3.1-8b-instant"
    TEMP = 0.0
    TICKERS = 10
    RUN_NAME = f"Run4_Clean_{MODEL}_T{TEMP}_{TICKERS}Assets"
    
    execute_experiment(MODEL, TEMP, TICKERS, RUN_NAME)