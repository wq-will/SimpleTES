EVAL_CONFIGS = [
    {"name": "a", "bs": 2, "dim": 128, "hiddendim": 128, "nomask": True, "seqlen": 256, "distribution": "normal", "seed": 17717},
    {"name": "b", "bs": 1, "dim": 128, "hiddendim": 128, "nomask": True, "seqlen": 768, "distribution": "cauchy", "seed": 17718},
    {"name": "c", "bs": 2, "dim": 384, "hiddendim": 128, "nomask": False, "seqlen": 256, "distribution": "normal", "seed": 17719},
    {"name": "d", "bs": 1, "dim": 128, "hiddendim": 128, "nomask": True, "seqlen": 512, "distribution": "normal", "seed": 17720},
    {"name": "e", "bs": 1, "dim": 128, "hiddendim": 128, "nomask": True, "seqlen": 1024, "distribution": "cauchy", "seed": 17721},
    {"name": "f", "bs": 1, "dim": 384, "hiddendim": 128, "nomask": False, "seqlen": 768, "distribution": "normal", "seed": 17722},
    {"name": "g", "bs": 1, "dim": 384, "hiddendim": 128, "nomask": True, "seqlen": 1024, "distribution": "normal", "seed": 17723},
]

EVAL_TOLERANCE = 2e-2
EVAL_SEED_TRIALS = 3
EVAL_SEED_STEP = 7
