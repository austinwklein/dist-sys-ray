# GenAI (ChatGPT GPT-5) used for function generation, code reviewed by Austin Klein
import random
import ray

def estimate_pi(num_samples: int) -> float:
    """Basic single-process Monte Carlo π estimation."""
    inside = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1:
            inside += 1
    return 4 * inside / num_samples


# Ray version of the same function as a remote, ready for distribution
@ray.remote
def estimate_pi_distributed(num_samples: int) -> float:
    return estimate_pi(num_samples)
