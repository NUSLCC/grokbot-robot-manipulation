"""openpi pi0.5 server-side checks (openpi .venv, Python 3.11). Read-only diagnostics; modes:

  files               no JAX: every checkpoint file hashed twice (crc32c) and compared with the gs://openpi-assets
                       object list (size + crc32c) and with any pi05_libero.bad* copy next to it. Prints
                       FILES_GCS_MISMATCH, FILES_READS_MATCH (0 = same file, different bytes: host RAM / disk),
                       FILES_BAD_COPY_DIFFERS.
  scan FP.json         CPU only (run with JAX_PLATFORMS=cpu): read the cached checkpoint params from disk TWICE with
                       orbax/tensorstore into host numpy (DIAG_READS times, default 3); per-leaf non-finite counts + crc32. Prints PARAMS_NONFINITE,
                       PARAMS_READS_MATCH (0 = two reads of the same files differ: host RAM / disk read problem) and
                       the leaves holding non-finite values. Saves the crc32s to FP.json.
  gpu FP.json [OBS]    one GPU (set CUDA_VISIBLE_DEVICES): VRAM pattern test, fp32/bf16 matmul vs a CPU reference,
                       restore the params ONTO the GPU and compare bit-for-bit with the CPU read, then load the policy
                       (pi05_libero, like serve_policy.py) and infer on openpi's example + the real obs.
                       Keys are prefixed GPU<id>_ (MEM_ERRORS, MATMUL_OK, RESTORE_MATCH, ACTIONS_FINITE).
  cpu [OBS]            policy inference on the CPU (JAX_PLATFORMS=cpu; slow) -> CPU_ACTIONS_FINITE.
"""
import importlib.metadata as md
import json
import os
import sys
import zlib

import numpy as np


def v(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return "-"


def ckpt_dir():
    from openpi.shared import download
    return download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")


def read_params(restore_type):
    from openpi.models import model as _model
    return _model.restore_params(os.path.join(str(ckpt_dir()), "params"), restore_type=restore_type)


def leaves_with_path(tree):
    import jax
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [(jax.tree_util.keystr(p), x) for p, x in flat]


def crc(a):
    a = np.ascontiguousarray(np.asarray(a))
    return zlib.crc32(a.view(np.uint8).reshape(-1)) & 0xFFFFFFFF


def nonfinite(a):
    return int((~np.isfinite(np.asarray(a, dtype=np.float32))).sum())


def stats(name, a):
    a = np.asarray(a, dtype=np.float64)
    fin = bool(np.all(np.isfinite(a)))
    print("%s shape=%s finite=%d nonfinite=%d absmax=%.4g" % (name, a.shape, fin, int((~np.isfinite(a)).sum()), np.nanmax(np.abs(a))))
    return fin


def versions():
    import jax
    print("SERVER python=%s " % sys.version.split()[0] + " ".join("%s=%s" % (p, v(p)) for p in (
        "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "flax", "orbax-checkpoint", "tensorstore", "numpy",
        "nvidia-cudnn-cu12", "nvidia-cublas-cu12")))
    print("SERVER backend=%s devices=%s CUDA_VISIBLE_DEVICES=%s XLA_FLAGS=%s" % (
        jax.default_backend(), jax.devices(), os.environ.get("CUDA_VISIBLE_DEVICES", ""), os.environ.get("XLA_FLAGS", "")))


GCS_LIST = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_libero/&fields=items(name,size,crc32c),nextPageToken"


def crc32c_file(path):
    """base64 crc32c like GCS, or a sha256 prefix when no crc32c module is installed (then no GCS compare)."""
    import base64
    import hashlib
    try:
        import google_crc32c
        h = google_crc32c.Checksum()
        upd, dig = h.update, lambda: base64.b64encode(h.digest()).decode()
    except ImportError:
        h = hashlib.sha256()
        upd, dig = h.update, lambda: "sha256:" + h.hexdigest()[:16]
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            upd(b)
    return dig()


def local_files(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = p
    return out


def mode_files():
    import glob
    import urllib.request
    root = str(ckpt_dir()).rstrip("/")
    print("FILES checkpoint dir: %s" % root)
    gcs = {}
    try:
        tok = ""
        while True:
            d = json.load(urllib.request.urlopen(GCS_LIST + ("&pageToken=" + tok if tok else ""), timeout=30))
            for it in d.get("items", []):
                gcs[it["name"].split("checkpoints/pi05_libero/", 1)[1]] = (int(it["size"]), it["crc32c"])
            tok = d.get("nextPageToken")
            if not tok:
                break
    except Exception as e:  # offline: still do the twice-read check
        print("FILES gs://openpi-assets object list not reachable (%s): no GCS compare" % e)
    loc = local_files(root)
    h1 = {r: crc32c_file(p) for r, p in sorted(loc.items())}
    h2 = {r: crc32c_file(p) for r, p in sorted(loc.items())}
    unstable = [r for r in h1 if h1[r] != h2[r]]
    print("FILES local=%d (%.2f GB) gcs=%d hash=%s" % (len(loc), sum(os.path.getsize(p) for p in loc.values()) / 1e9, len(gcs),
                                                      "crc32c" if not next(iter(h1.values()), "").startswith("sha256:") else "sha256 (pip google-crc32c for a GCS compare)"))
    if gcs:
        missing = [r for r in gcs if r not in loc]
        bad = [r for r in gcs if r in loc and (os.path.getsize(loc[r]) != gcs[r][0] or (not h1[r].startswith("sha256:") and h1[r] != gcs[r][1]))]
        for r in missing[:5]:
            print("FILES missing locally: %s" % r)
        for r in bad[:5]:
            print("FILES differs from GCS: %s size=%d/%d crc32c=%s/%s" % (r, os.path.getsize(loc[r]), gcs[r][0], h1[r], gcs[r][1]))
        print("FILES_GCS_MISMATCH=%d" % (len(missing) + len(bad)))
    for r in unstable[:5]:
        print("FILES read twice, different bytes: %s" % r)
    print("FILES_READS_MATCH=%d" % int(not unstable))
    diff_total = 0
    for other in sorted(glob.glob(root + ".bad*")):
        ol = local_files(other)
        diffs = [r for r in h1 if r not in ol or os.path.getsize(ol[r]) != os.path.getsize(loc[r]) or crc32c_file(ol[r]) != h1[r]]
        diff_total += len(diffs)
        print("FILES vs %s: %d of %d files differ %s" % (other, len(diffs), len(h1), " ".join(diffs[:3])))
    print("FILES_BAD_COPY_DIFFERS=%d" % diff_total)


def mode_scan(fp_path):
    versions()
    reads = []
    for r in range(max(2, int(os.environ.get("DIAG_READS", "3")))):
        rows = [(k, str(np.asarray(x).dtype), int(np.size(x)), nonfinite(x), crc(x)) for k, x in leaves_with_path(read_params(np.ndarray))]
        reads.append(rows)
        print("PARAMS read %d: leaves=%d values=%d nonfinite=%d" % (r + 1, len(rows), sum(x[2] for x in rows), sum(x[3] for x in rows)))
    differ = sorted({a[0] for rr in reads[1:] for a, b in zip(reads[0], rr) if a[4] != b[4]})
    match = not differ
    print("PARAMS leaves that differ between reads: %d %s" % (len(differ), " ".join(differ[:5])))
    bad = sorted([x for x in reads[0] if x[3]], key=lambda x: -x[3])
    for k, dt, n, nf, _ in bad[:8]:
        print("PARAMS nonfinite leaf %s dtype=%s size=%d nonfinite=%d" % (k, dt, n, nf))
    print("PARAMS_NONFINITE=%d" % sum(x[3] for x in reads[0]))
    print("PARAMS_NONFINITE_LEAVES=%d" % len(bad))
    print("PARAMS_READS_MATCH=%d" % int(match))
    json.dump({k: c for k, _, _, _, c in reads[0]}, open(fp_path, "w"))


def infer(tag, obs_path):
    from openpi.policies import libero_policy, policy_config
    from openpi.training import config as _config
    policy = policy_config.create_trained_policy(_config.get_config("pi05_libero"), ckpt_dir())
    np.random.seed(0)
    fin = stats("%s example actions" % tag, policy.infer(libero_policy.make_libero_example())["actions"])
    if obs_path and os.path.isfile(obs_path):
        d = np.load(obs_path)
        el = {"observation/image": d["image"], "observation/wrist_image": d["wrist_image"],
              "observation/state": d["state"], "prompt": str(d["prompt"])}
        for i in range(2):
            a = np.asarray(policy.infer(el)["actions"])
            fin = stats("%s real-obs actions (call %d)" % (tag, i + 1), a) and fin
        print("%s real-obs first action=%s" % (tag, np.array2string(a[0], precision=4)))
    print("%s_ACTIONS_FINITE=%d" % (tag, int(fin)))


def mode_gpu(fp_path, obs_path):
    import jax
    import jax.numpy as jnp
    versions()
    tag = "GPU%s" % os.environ.get("CUDA_VISIBLE_DEVICES", "x").split(",")[0]
    dev = jax.devices()[0]
    # 1) VRAM pattern test: fill most free memory with known patterns, read back, count wrong words.
    limit = (dev.memory_stats() or {}).get("bytes_limit", 8 << 30)
    chunk = 256 << 20; n = max(1, int(limit * 0.8) // chunk); words = chunk // 4
    gen = jax.jit(lambda i, s: (jnp.arange(words, dtype=jnp.uint32) * jnp.uint32(2654435761)) ^ (i * jnp.uint32(0x9E3779B9) + s))
    chk = jax.jit(lambda x, i, s: jnp.sum(x != gen(i, s), dtype=jnp.uint32))
    errs = 0
    for s in (np.uint32(0), np.uint32(0xFFFFFFFF)):
        bufs = [gen(np.uint32(i), s) for i in range(n)]
        jax.block_until_ready(bufs)
        errs += sum(int(chk(b, np.uint32(i), s)) for i, b in enumerate(bufs))
        del bufs
    print("%s memtest: %d x 256 MiB x 2 patterns" % (tag, n))
    print("%s_MEM_ERRORS=%d" % (tag, errs))
    # 2) matmul vs CPU reference (fp32 HIGHEST and bf16), repeated: wrong or non-repeatable results = bad GPU math.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4096, 4096), dtype=np.float32); b = rng.standard_normal((4096, 4096), dtype=np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64); scale = np.abs(ref).max()
    f32 = jax.jit(lambda x, y: jnp.matmul(x, y, precision=jax.lax.Precision.HIGHEST))
    b16 = jax.jit(lambda x, y: jnp.matmul(x.astype(jnp.bfloat16), y.astype(jnp.bfloat16), preferred_element_type=jnp.float32))
    outs = [np.asarray(f32(a, b)) for _ in range(5)]
    e32 = max(float(np.abs(o - ref).max() / scale) for o in outs); rep = all(np.array_equal(outs[0], o) for o in outs)
    e16 = float(np.abs(np.asarray(b16(a, b)) - ref).max() / scale)
    ok = e32 < 1e-4 and e16 < 5e-2 and rep and all(np.isfinite(o).all() for o in outs)
    print("%s matmul fp32 relerr=%.2e repeatable=%d bf16 relerr=%.2e" % (tag, e32, int(rep), e16))
    print("%s_MATMUL_OK=%d" % (tag, int(ok)))
    # 3) restore the checkpoint onto this GPU and compare bit-for-bit with the CPU read (scan mode).
    if fp_path and os.path.isfile(fp_path):
        ref_crc = json.load(open(fp_path))
        mism = []; nf = 0
        for k, x in leaves_with_path(read_params(jax.Array)):
            h = np.asarray(jax.device_get(x)); nf += nonfinite(h)
            if ref_crc.get(k) != crc(h):  # same dtype as the CPU read (no dtype= cast), one replicated device
                mism.append(k)
            del x, h
        print("%s restore on GPU: mismatched leaves=%d nonfinite=%d %s" % (tag, len(mism), nf, " ".join(mism[:5])))
        print("%s_RESTORE_MATCH=%d" % (tag, int(not mism)))
    # 4) the policy itself on this GPU.
    infer(tag, obs_path)


def main(argv):
    mode = argv[1] if len(argv) > 1 else "gpu"
    if mode == "files":
        mode_files()
    elif mode == "scan":
        mode_scan(argv[2])
    elif mode == "gpu":
        mode_gpu(argv[2] if len(argv) > 2 else "", argv[3] if len(argv) > 3 else "")
    elif mode == "cpu":
        versions(); infer("CPU", argv[2] if len(argv) > 2 else "")
    else:
        print(__doc__); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
