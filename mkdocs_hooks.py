"""MkDocs hooks for the docs site (mkdocs.yml: hooks).

The repo keeps README.md and TROUBLESHOOTING.md at the root (where GitHub shows them) and the rest in docs/.
MkDocs only reads docs/, so this hook adds the two root files to the site (README.md as the home page) and
rewrites relative links so the same Markdown works on GitHub and on the site:
  * links between the pages above point to the site pages (docs/WSL2.md -> WSL2.md, ../README.md -> index.md),
  * links to any other repo file (scripts/..., tests/..., mkdocs.yml) point to that file on GitHub.
"""
import logging
import os
import posixpath
import re

from mkdocs.structure.files import File

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_BLOB = "https://github.com/NUSLCC/grokbot-robot-manipulation/blob/{branch}/{path}"
ROOT_PAGES = {"README.md": "index.md", "TROUBLESHOOTING.md": "TROUBLESHOOTING.md"}
_ORIGIN = {}  # site src_uri -> repo path
LOG = logging.getLogger("mkdocs.hooks.repo_links")  # a warning here fails `mkdocs build --strict`
_LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def _branch(config):
    return (config.get("extra") or {}).get("repo_branch", "main")


def on_files(files, config):
    for repo_path, site_path in ROOT_PAGES.items():
        with open(os.path.join(ROOT, repo_path), encoding="utf-8") as fh:
            files.append(File.generated(config, site_path, content=fh.read()))
        _ORIGIN[site_path] = repo_path
    for f in files.documentation_pages():
        _ORIGIN.setdefault(f.src_uri, "docs/" + f.src_uri)
    return files


def _rewrite(target, page_repo_path, page_site_path, branch):
    if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith(("#", "/")):
        return target
    path, _, anchor = target.partition("#")
    repo = posixpath.normpath(posixpath.join(posixpath.dirname(page_repo_path), path))
    if repo in ROOT_PAGES:
        site = ROOT_PAGES[repo]
    elif repo.startswith("docs/") and repo.endswith(".md"):
        site = repo[len("docs/"):]
    else:
        if repo.startswith("..") or not os.path.exists(os.path.join(ROOT, repo)):
            LOG.warning("%s links to '%s', which is not a file in this repo", page_repo_path, target)
        url = REPO_BLOB.format(branch=branch, path=repo)
        return url + ("#" + anchor if anchor else "")
    rel = posixpath.relpath(site, posixpath.dirname(page_site_path) or ".")
    return rel + ("#" + anchor if anchor else "")


def on_page_markdown(markdown, page, config, files):
    site_path = page.file.src_uri
    repo_path = _ORIGIN.get(site_path, "docs/" + site_path)
    branch = _branch(config)
    out, in_code = [], False
    for line in markdown.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        if not in_code:
            line = _LINK.sub(lambda m: m.group(1) + _rewrite(m.group(2), repo_path, site_path, branch) + m.group(3), line)
        out.append(line)
    return "\n".join(out)
