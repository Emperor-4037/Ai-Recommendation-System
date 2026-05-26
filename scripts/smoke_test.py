import subprocess
import time
import requests
import sys

def main():
    print("--- Starting Smoke Test ---")
    
    # 1. Start FastAPI server in background
    print("Starting FastAPI server...")
    server_process = subprocess.Popen(
        ["uvicorn", "recsys.serving.api:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to be ready
    max_retries = 10
    ready = False
    for i in range(max_retries):
        try:
            resp = requests.get("http://127.0.0.1:8000/ready")
            if resp.status_code == 200:
                ready = True
                break
        except requests.exceptions.ConnectionError:
            pass
        print(f"Waiting for server... ({i+1}/{max_retries})")
        time.sleep(2)
        
    if not ready:
        print("Error: Server failed to start or load models.")
        server_process.terminate()
        sys.exit(1)
        
    print("Server is ready and models are loaded.")
    
    # 2. Test Sync Mode
    print("Testing /recommend/sync...")
    sync_payload = {"user_id": 1, "num_candidates": 5, "user_intent": "casual"}
    resp_sync = requests.post("http://127.0.0.1:8000/recommend/sync", json=sync_payload)
    
    if resp_sync.status_code == 200:
        print("Sync mode success.")
        print(f"Candidates returned: {len(resp_sync.json()['candidates'])}")
    else:
        print(f"Sync mode failed: {resp_sync.text}")
        server_process.terminate()
        sys.exit(1)
        
    # 3. Test Spark Mode
    print("Testing /recommend/spark...")
    spark_payload = {"user_id": 1, "num_candidates": 5, "user_intent": "long_term"}
    resp_spark = requests.post("http://127.0.0.1:8000/recommend/spark", json=spark_payload)
    
    if resp_spark.status_code == 200:
        print("Spark mode success.")
        print(f"Candidates returned: {len(resp_spark.json()['candidates'])}")
    else:
        print(f"Spark mode failed: {resp_spark.text}")
        server_process.terminate()
        sys.exit(1)
        
    # Teardown
    print("--- Smoke Test Passed! ---")
    server_process.terminate()

if __name__ == "__main__":
    main()
