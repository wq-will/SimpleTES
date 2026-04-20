EVAL_CONFIGS = [
    {"name": "a", "batch_size": 2, "image_size": 224, "patch_size": 16, "num_classes": 10, "dim": 512, "depth": 6, "heads": 8, "mlp_dim": 2048, "channels": 3, "dropout": 0.0, "emb_dropout": 0.0, "seed": 17717}
]

EVAL_TOLERANCE = 1e-2
EVAL_SEED_TRIALS = 2
EVAL_SEED_STEP = 7