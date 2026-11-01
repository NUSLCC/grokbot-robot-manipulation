#!/usr/bin/env bash
# Docs: every relative Markdown link in README.md, TROUBLESHOOTING.md and docs/*.md points to a file that exists,
# every page in mkdocs.yml's nav exists, and docs/VALIDATION.md keeps its required sections. (CI has no mkdocs;
# `mkdocs build --strict` additionally checks anchors, see docs/VALIDATION.md "Docs site".)
set -uo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
python3 - "$K" <<'PY'
import os, re, sys
k = sys.argv[1]; bad = 0; n = 0
def ok(msg): global n; n += 1; print("ok   - " + msg)
def fail(msg): global bad; bad += 1; print("FAIL - " + msg)
pages = ["README.md", "TROUBLESHOOTING.md"] + ["docs/" + f for f in sorted(os.listdir(os.path.join(k, "docs"))) if f.endswith(".md")]
link = re.compile(r"\]\(([^)\s]+)\)")
for p in pages:
    code = False; missing = []
    for line in open(os.path.join(k, p), encoding="utf-8"):
        if line.lstrip().startswith(("```", "~~~")): code = not code
        if code: continue
        for t in link.findall(line):
            if re.match(r"^[a-z][a-z0-9+.-]*:", t) or t.startswith("#"): continue
            path = t.split("#", 1)[0]
            if not os.path.exists(os.path.normpath(os.path.join(k, os.path.dirname(p), path))): missing.append(t)
    ok(p + ": relative links resolve") if not missing else fail(p + ": missing link targets " + ", ".join(missing))
nav = re.findall(r"^\s+- [^:]+: (\S+\.md)\s*$", open(os.path.join(k, "mkdocs.yml")).read(), re.M)
roots = {"index.md": "README.md", "TROUBLESHOOTING.md": "TROUBLESHOOTING.md"}
miss = [x for x in nav if not os.path.exists(os.path.join(k, roots.get(x, "docs/" + x)))]
ok("mkdocs.yml nav (%d pages) exists" % len(nav)) if nav and not miss else fail("mkdocs nav missing: %s" % miss)
v = open(os.path.join(k, "docs/VALIDATION.md"), encoding="utf-8").read()
for h in ["## The runner", "## The 14 steps", "## Quick run, dual RTX 3090", "## Full-length run, dual RTX 3090",
          "## Issues found and fixed during validation", "## How to run", "## Machine profiles", "## Docs site"]:
    ok("VALIDATION.md has '%s'" % h) if ("\n" + h + "\n") in v else fail("VALIDATION.md lacks '%s'" % h)
steps = re.findall(r'^add (\w+) ', open(os.path.join(k, "scripts/run_wsl_headless.sh")).read(), re.M)
miss = [s for s in steps if "`%s`" % s not in v]
ok("VALIDATION.md documents all %d runner steps" % len(steps)) if len(steps) == 14 and not miss else fail("steps %d, undocumented %s" % (len(steps), miss))
ok("README links docs/VALIDATION.md") if "](docs/VALIDATION.md)" in open(os.path.join(k, "README.md")).read() else fail("README lacks VALIDATION link")
print(); print("ALL PASS" if not bad else "%d FAILED" % bad); sys.exit(bad)
PY
