import mlflow
import subprocess
import json
import os

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
        
        # Save the actual debate JSON as an artifact.
        mlflow.log_artifact(f"results/{run_name}.json")
        mlflow.log_artifact("evaluation_data.csv")

        # 7. Execute Portfolio Backtest (Sharpe, MDD, vs Buy-and-Hold)
        print("\n📈 Executing Portfolio Backtest...")
        try:
            subprocess.run(["python", "backtest.py"], check=True)
            mlflow.log_artifact(f"results/{run_name}_backtest.json")
        except Exception as e:
            print(f"⚠️ Backtest step failed (swarm results still saved): {e}")
        
        print(f"\n✅ Experiment Logged Successfully. Accuracy: {accuracy}%")

if __name__ == "__main__":
    MODEL = "llama-3.3-70b-versatile"
    TEMP = 0.5
    TICKERS = 5

    os.environ["EXPERIMENT_MODEL"] = MODEL
    os.environ["EXPERIMENT_TEMP"] = str(TEMP)


    RUN_NAME = f"Run3_{MODEL}_T{TEMP}_{TICKERS}Assets"
    os.environ["RUN_NAME"] = RUN_NAME
    execute_experiment(MODEL, TEMP, TICKERS, RUN_NAME) 