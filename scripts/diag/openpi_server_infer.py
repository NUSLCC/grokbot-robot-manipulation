"""openpi pi0.5 server-side check (runs in the openpi .venv, Python 3.11). Read-only diagnostics.

    python openpi_server_infer.py [OBS.npz]

Prints jax/jaxlib/flax/orbax/numpy versions, JAX backend + devices + XLA env vars, scans the cached checkpoint params
for NaN/Inf (on the host, no GPU), loads the policy like serve_policy.py (pi05_libero) and runs policy.infer on
(a) openpi's own random LIBERO example and (b) the real first observation from OBS.npz, printing action stats.
Run it twice to compare: on the GPU (default) and with JAX_PLATFORMS=cpu.
"""
import os
import sys
import importlib.metadata as md

import numpy as np


def v(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return "-"


def stats(name, a):
    a = np.asarray(a, dtype=np.float64)
    fin = bool(np.all(np.isfinite(a)))
    print("%s shape=%s finite=%d nonfinite=%d absmax=%.4g mean=%.4g" % (
        name, a.shape, fin, int((~np.isfinite(a)).sum()), np.nanmax(np.abs(a)), np.nanmean(a)))
    return fin


def main(argv):
    import jax
    tag = jax.default_backend().upper()
    print("SERVER python=%s " % sys.version.split()[0] + " ".join(
        "%s=%s" % (p, v(p)) for p in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "flax", "orbax-checkpoint",
                                     "numpy", "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "openpi", "openpi-client")))
    print("SERVER backend=%s devices=%s" % (jax.default_backend(), jax.devices()))
    print("SERVER env " + " ".join("%s=%s" % (k, os.environ.get(k, "")) for k in (
        "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE",
        "JAX_DEFAULT_MATMUL_PRECISION", "OPENPI_DATA_HOME")))
    from openpi.policies import libero_policy, policy_config
    from openpi.shared import download
    from openpi.training import config as _config
    from openpi.models import model as _model
    cfg = _config.get_config("pi05_libero")
    ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    print("SERVER checkpoint=%s" % ckpt)
    if os.environ.get("DIAG_SCAN_PARAMS", "1") == "1" and tag != "CPU":
        params = _model.restore_params(os.path.join(str(ckpt), "params"), restore_type=np.ndarray)
        leaves = jax.tree_util.tree_leaves(params)
        bad = sum(int((~np.isfinite(np.asarray(x, dtype=np.float32))).sum()) for x in leaves)
        print("PARAMS leaves=%d values=%d nonfinite=%d" % (len(leaves), sum(int(np.size(x)) for x in leaves), bad))
        print("PARAMS_NONFINITE=%d" % bad)
        del params, leaves
    policy = policy_config.create_trained_policy(cfg, ckpt)
    np.random.seed(0)
    fin = stats("%s example actions" % tag, policy.infer(libero_policy.make_libero_example())["actions"])
    if len(argv) > 1 and os.path.isfile(argv[1]):
        d = np.load(argv[1])
        el = {"observation/image": d["image"], "observation/wrist_image": d["wrist_image"],
              "observation/state": d["state"], "prompt": str(d["prompt"])}
        a = np.asarray(policy.infer(el)["actions"])
        fin = stats("%s real-obs actions" % tag, a) and fin
        print("%s real-obs first action=%s" % (tag, np.array2string(a[0], precision=4)))
    print("%s_ACTIONS_FINITE=%d" % (tag if tag == "CPU" else "GPU", int(fin)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
