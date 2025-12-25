import os
import time
from multiprocessing import Pool
from PIL import Image
import numpy as np

from filters import (
    apply_grayscale,
    apply_gaussian_blur,
    apply_sobel,
    apply_sharpen,
    apply_brightness
)

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"


def process_single_image(args):
    img_path, output_path = args

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    img_np = apply_grayscale(img_np)
    img_np = apply_gaussian_blur(img_np)
    img_np = apply_sobel(img_np)
    img_np = apply_sharpen(img_np)
    img_np = apply_brightness(img_np, factor=1.2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(img_np).save(output_path)


def get_all_image_tasks():
    tasks = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, INPUT_DIR)
                output_path = os.path.join(OUTPUT_DIR, relative_path)
                tasks.append((input_path, output_path))
    return tasks


def run_experiment(num_processes):
    tasks = get_all_image_tasks()

    start = time.time()
    with Pool(processes=num_processes) as pool:
        pool.map(process_single_image, tasks)
    end = time.time()

    return end - start, len(tasks)


if __name__ == "__main__":
    process_counts = [1, 2, 4]
    baseline_time = None

    print("Workers (P) | Execution time | Speed-up | Efficiency")
    print("-" * 55)

    for p in process_counts:
        exec_time, total_images = run_experiment(p)

        if p == 1:
            baseline_time = exec_time
            speedup = 1.0
            efficiency = 1.0
        else:
            speedup = baseline_time / exec_time
            efficiency = speedup / p

        print(f"{p:<11} | {exec_time:.4f} s      | {speedup:.4f} | {efficiency*100:.2f}%")
