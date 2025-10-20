# GenAI (ChatGPT GPT-5) used for boilerplate generation, code reviewed by Austin Klein
import sys
import time
import ray
import os  # For CPU core detection and setting env variable for running in VM (not strictly required by Ray)
from monte_carlo_pi import estimate_pi, estimate_pi_distributed

# For running in VM with no dedicated GPU
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
SAMPLES = 100_000_000

def run_static(NUM_SAMPLES :int):
    """Run the Monte Carlo π estimation in a single process."""
    print("Running Monte Carlo π estimation (static, single process)...")
    start = time.time()
    pi_estimate = estimate_pi(NUM_SAMPLES)
    end = time.time()
    print(f"Estimated π = {pi_estimate}")
    print(f"Time taken: {end - start:.3f} seconds\n")


def run_distributed(NUM_SAMPLES :int):
    """Run the Monte Carlo π estimation distributed across multiple workers."""
    print("Running Monte Carlo π estimation (distributed with Ray)...")
    ray.init(ignore_reinit_error=True)

    import os
    NUM_TASKS = os.cpu_count()

    print(f"Detected {NUM_TASKS} CPU cores - runnning {NUM_TASKS} parallel tasks")

    start = time.time()
    # Split work across tasks
    tasks = [estimate_pi_distributed.remote(NUM_SAMPLES // NUM_TASKS) for _ in range(NUM_TASKS)]
    results = ray.get(tasks)
    pi_estimate = sum(results) / len(results)
    end = time.time()

    print(f"Estimated π = {pi_estimate}")
    print(f"Time taken: {end - start:.3f} seconds\n")

    ray.shutdown()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ray_demo.py mc --static|--distributed")
        sys.exit(1)

    mode = sys.argv[1]
    method = sys.argv[2]

    if mode == "mc":
        if method == "--static":
            run_static(NUM_SAMPLES=SAMPLES)
        elif method == "--distributed":
            run_distributed(NUM_SAMPLES=SAMPLES)
        else:
            print("Invalid method. Use --static or --distributed.")
    else:
        print("Unknown mode. Currently only supports 'mc'.")
