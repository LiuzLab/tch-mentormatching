import os
import json
import hashlib

CACHE_DIR = "cv_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_key(file, num_candidates):
    # Create a unique cache key based on file content and number of candidates
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)  # Reset file pointer
    return f"{file_hash}_{num_candidates}"

def save_to_cache(key, data):
    with open(os.path.join(CACHE_DIR, f"{key}.json"), 'w') as f:
        json.dump(data, f)

def load_from_cache(key):
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None