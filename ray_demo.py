# GenAI (ChatGPT GPT-5) used for boilerplate generation, code reviewed by Austin Klein
import sys
import time
import ray
import os  # For CPU core detection and setting env variable for running in VM (not strictly required by Ray)
from monte_carlo_pi import estimate_pi, estimate_pi_distributed

# For running in VM with no dedicated GPU
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
SAMPLES = 200_000_000
instance = [0]

def run_static(NUM_SAMPLES: int):
    """Run the Monte Carlo π estimation in a single process."""
    print("Running Monte Carlo π estimation (static, single process)...")
    start = time.time()
    pi_estimate = estimate_pi(NUM_SAMPLES)
    end = time.time()
    print(f"Estimated π = {pi_estimate}")
    print(f"Time taken: {end - start:.3f} seconds\n")


def run_distributed(NUM_SAMPLES: int, instance):
    instance[0] = instance[0] + 1

    """Run the Monte Carlo π estimation distributed across multiple workers."""
    print("Running Monte Carlo π estimation (distributed with Ray)...")

    # Not worried about input validation here
    user_input = input("Choose a process multiplier or press enter to use default of none (1)... ")
    MULTIPLIER = int(user_input) if user_input and user_input != "0" else 1

    NUM_TASKS = os.cpu_count()
    print(f"Detected {NUM_TASKS} CPU cores - running {NUM_TASKS*MULTIPLIER} tasks with a multiplier of {MULTIPLIER}...")

    start = time.time()
    # Split work across tasks
    tasks = [estimate_pi_distributed.options(name=f"Monte-Carlo-Estimation_{instance[0]}").remote(NUM_SAMPLES // NUM_TASKS) for _ in range(NUM_TASKS*MULTIPLIER)]
    results = ray.get(tasks)
    pi_estimate = sum(results) / len(results)
    end = time.time()

    print(f"Estimated π = {pi_estimate}")
    print(f"Time taken: {end - start:.3f} seconds\n")


def interactive_runner():
    """Interactive loop: lets you run tasks, inspect the dashboard, or run static simulation."""

    # Start Ray once at the beginning with dashboard
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=True,
            dashboard_host="0.0.0.0"
        )

    while True:
        print("\nOptions:")
        print("1: Run static Monte Carlo π estimation (single process)")
        print("2: Run distributed Monte Carlo π estimation with Ray")
        print("3: Exit")

        choice = input("Enter option number: ").strip()
        if choice == "1":
            run_static(SAMPLES)
        elif choice == "2":
            run_distributed(SAMPLES, instance=instance)
        elif choice == "3":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

    # Shut down Ray after user exits
    ray.shutdown()
    print("Ray instance has been shut down.")


if __name__ == "__main__":
    interactive_runner()
