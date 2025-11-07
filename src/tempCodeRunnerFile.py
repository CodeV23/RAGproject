        if result_keys:
            print(f"[INFO] Found {len(result_keys)} matching buckets via LSH → {result_keys[:k]}")
        else:
            print("[WARN] No matches found via LSH; returning top-1 random chunk.")