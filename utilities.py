import multiprocessing
import psutil

async def choose_optimal_batch_size(
    total_items: int,
    safety_factor: float = 0.8  # Reserve 20% memory for other processes
) -> int:
    """
    Dynamically computes batch size based on:
    - CPU cores (parallelism)
    - Available memory (RAM constraints)
    - Total workload size
    """
    cpu_count = multiprocessing.cpu_count()
    mem_info = psutil.virtual_memory()

    # Memory-based constraints (adjust based on your task's memory usage per item)
    BYTES_PER_ITEM = 1024  # Example: 1KB per item (replace with actual estimate)
    max_mem_batch = int((mem_info.available * safety_factor) / BYTES_PER_ITEM)

    # CPU-based constraints
    ideal_concurrent_batches = cpu_count * 2  # Adjust multiplier for I/O vs. CPU bound
    cpu_based_batch = max(1, total_items // ideal_concurrent_batches)

    # Combine constraints
    max_batch_size = min(max_mem_batch, cpu_based_batch)

    # Ensure batch is neither too small nor too large
    min_batch_size = max(1, total_items // 100)  # At least 1% of total items
    optimal_batch_size = max(min_batch_size, min(max_batch_size, total_items))

    return optimal_batch_size

def format_invalid_movies(invalid_rated_movies):
    filtered = [m for m in invalid_rated_movies if m]
    formatted = []
    for i, movie_list in enumerate(filtered):
        lines = [f"{title}: {rating}" for title, rating in movie_list]
        formatted.append("\n".join(lines))
        if i < len(filtered) - 1:
            formatted.append("")  # double newline
    return "\n\n".join(formatted)
