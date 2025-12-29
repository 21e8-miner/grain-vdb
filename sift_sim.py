def make_sift_dataset(N, D, seed=42):
    # Simulate SIFT-like local features (integer-based descriptors normalized)
    rng = np.random.default_rng(seed)
    # SIFT descriptors are typically 128D, often gradients in 8 directions x 4x4 grid
    # We simulate this by generating clustered positive values
    raw = rng.integers(0, 255, size=(N, D), dtype=np.uint8).astype(np.float32)
    # Norm L2
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / (norm + 1e-9)

