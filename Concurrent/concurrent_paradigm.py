#Main code for concurrent.futures
import os
import time
import concurrent.futures
from pathlib import Path
from PIL import Image
import numpy as np
import filters

def process_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        # Pipeline: Applying all 5 operations
        img_np = filters.apply_grayscale(img_np)
        img_np = filters.apply_gaussian_blur(img_np)
        img_np = filters.apply_sobel(img_np)
        img_np = filters.apply_sharpen(img_np)
        img_np = filters.apply_brightness(img_np)
        
        # Path Logic:
        # Image: food/input/apple_pie/101.jpg
        # Save to food/output/apple_pie/101.jpg
        relative_path = image_path.relative_to("food/input")
        output_path = Path("food/output") / relative_path
        
        # Create the subfolder (e.g., food/output/apple_pie) if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(img_np).save(output_path)
        return True
    except Exception as e:
        return f"Error {image_path}: {e}"

if __name__ == "__main__":
    # Pointing to the new folder structure
    input_base = Path("food/input")
    image_paths = list(input_base.rglob("*.jpg"))
    
    if not image_paths:
        print(f"No images found in {input_base}. Please check your folder structure.")
        exit()

    # Storage for performance data
    serial_time = 0.0
    worker_counts = [1, 2, 4]
    
    print(f"{'Workers (P)':<12} | {'Execution time':<14} | {'Speed-up':<12} | {'Efficiency':<12}")
    print("-" * 65)

    for p in worker_counts:
        start_time = time.time()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=p) as executor:
            #Cast to list to force the generator to execute
            list(executor.map(process_image, image_paths))
            
        parallel_run_time = time.time() - start_time
        
        if p == 1:
            serial_time = parallel_run_time
            speedup = 1.0
            efficiency = 1.0
        else:
            speedup = serial_time / parallel_run_time
            efficiency = speedup / p
            
        print(f"{p:<12} | {parallel_run_time:<14.4f} | {speedup:<12.4f} | {efficiency:<12.2%}")
