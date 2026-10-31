#!/usr/bin/env python3
"""Write LIBERO's config.yaml non-interactively (Python 3.8+, no third-party imports).

    python libero_config.py CONFIG_DIR

LIBERO (both the original Lifelong-Robot-Learning/LIBERO and the hf-libero wheel LeRobot uses) asks
"Do you want to specify a custom path for the dataset folder? (Y/N)" via input() the first time
`libero.libero` is imported and $LIBERO_CONFIG_PATH/config.yaml (default ~/.libero/config.yaml) is missing.
Under a headless runner stdin is closed, so that input() raises "EOFError: EOF when reading a line".

This script writes exactly what LIBERO writes when you answer "N" (get_default_path_dict()), for the libero
package of the Python that runs it, into CONFIG_DIR/config.yaml. It locates the package with
importlib.util.find_spec and never imports libero.libero (that import is what prompts).
An existing config is kept if its benchmark_root matches this package; otherwise it is backed up and rewritten
(keeping a custom 'datasets' path if that directory exists).
"""
import importlib.util
import json
import os
import sys

KEYS = ("benchmark_root", "bddl_files", "init_states", "datasets", "assets")


def find_benchmark_root():
    spec = importlib.util.find_spec("libero")
    if spec is None:
        return None
    for loc in list(spec.submodule_search_locations or []):
        cand = os.path.join(loc, "libero")
        if os.path.isfile(os.path.join(cand, "__init__.py")):
            # LIBERO uses os.path.dirname(os.path.abspath(libero/libero/__init__.py))
            return os.path.dirname(os.path.abspath(os.path.join(cand, "__init__.py")))
    return None


def default_paths(root):
    # Same strings as libero.libero.get_default_path_dict() (including the "./" segments).
    return {
        "benchmark_root": root,
        "bddl_files": os.path.join(root, "./bddl_files"),
        "init_states": os.path.join(root, "./init_files"),
        "datasets": os.path.join(root, "../datasets"),
        "assets": os.path.join(root, "./assets"),
    }


def read_simple_yaml(path):
    """Parse the flat 'key: value' file LIBERO's yaml.dump writes (plain or quoted scalars)."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] == '"':
                try:
                    v = json.loads(v)
                except ValueError:
                    v = v[1:-1]
            elif len(v) >= 2 and v[0] == v[-1] == "'":
                v = v[1:-1].replace("''", "'")
            out[k.strip()] = v
    return out


def write_yaml(path, cfg):
    # JSON-quoted strings are valid YAML scalars, so yaml.load(FullLoader) reads these back exactly.
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for k in KEYS:
            f.write("%s: %s\n" % (k, json.dumps(cfg[k])))
    os.replace(tmp, path)


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    cfg_dir = os.path.abspath(os.path.expanduser(argv[1]))
    root = find_benchmark_root()
    if root is None:
        print("libero_config: no 'libero' package importable by %s" % sys.executable, file=sys.stderr)
        return 3
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_file = os.path.join(cfg_dir, "config.yaml")
    want = default_paths(root)
    if os.path.isfile(cfg_file):
        try:
            have = read_simple_yaml(cfg_file)
        except OSError:
            have = {}
        if have.get("benchmark_root") == root and all(k in have for k in KEYS):
            print("libero_config: kept %s (benchmark_root %s)" % (cfg_file, root))
            return 0
        custom = have.get("datasets")
        if custom and os.path.isdir(custom):
            want["datasets"] = custom
        os.replace(cfg_file, cfg_file + ".bak")
        print("libero_config: previous config moved to %s.bak" % cfg_file)
    write_yaml(cfg_file, want)
    print("libero_config: wrote %s (benchmark_root %s)" % (cfg_file, root))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
