"""Replace the provisional paper link when the exact arXiv record is public."""

from __future__ import annotations

import argparse
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


TITLE = "SceneWeave: Audience-Guided Scene Exploration for Dramatic Adaptation"
AUTHORS = ["Aolin Li", "Xiao Xu", "Haihua Xie"]
ATOM = "{http://www.w3.org/2005/Atom}"
OLD_LINK = f"[{TITLE}](docs/SceneWeave.pdf)"
OLD_CITATION = """@unpublished{li2026sceneweave,
  author = {Li, Aolin and Xu, Xiao and Xie, Haihua},
  title = {SceneWeave: Audience-Guided Scene Exploration for Dramatic Adaptation},
  year = {2026},
  note = {Manuscript}
}"""
OLD_GUIDANCE = (
    "For the paper, use the authors and title below. Add the arXiv identifier once "
    "the public record is available. Cite the released models and experimental "
    "materials separately with their Zenodo DOI."
)
NEW_GUIDANCE = (
    "Cite the paper using its arXiv record. Cite the released models and "
    "experimental materials separately with their Zenodo DOI."
)


def find_paper(feed: bytes) -> str | None:
    root = ET.fromstring(feed)
    for entry in root.findall(f"{ATOM}entry"):
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        authors = [
            " ".join((author.findtext(f"{ATOM}name") or "").split())
            for author in entry.findall(f"{ATOM}author")
        ]
        if title != TITLE or authors != AUTHORS:
            continue
        record = entry.findtext(f"{ATOM}id") or ""
        match = re.fullmatch(r"https?://arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?", record)
        if match:
            return match.group(1)
    return None


def update_readme(readme: Path, arxiv_id: str) -> bool:
    original = readme.read_text(encoding="utf-8")
    url = f"https://arxiv.org/abs/{arxiv_id}"
    if f"[{TITLE}]({url})" in original:
        return False
    required = (OLD_LINK, OLD_CITATION, OLD_GUIDANCE)
    if any(original.count(part) != 1 for part in required):
        raise ValueError("README paper link or citation changed; review before updating")
    citation = f"""@misc{{li2026sceneweave,
  author = {{Li, Aolin and Xu, Xiao and Xie, Haihua}},
  title = {{{TITLE}}},
  year = {{2026}},
  eprint = {{{arxiv_id}}},
  archivePrefix = {{arXiv}},
  primaryClass = {{cs.CL}},
  url = {{{url}}}
}}"""
    revised = original.replace(OLD_LINK, f"[{TITLE}]({url})")
    revised = revised.replace(OLD_CITATION, citation)
    revised = revised.replace(OLD_GUIDANCE, NEW_GUIDANCE)
    readme.write_text(revised, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed-file", type=Path, help="Use a saved Atom feed for offline testing")
    parser.add_argument("--readme", type=Path, default=Path(__file__).resolve().parents[1] / "README.md")
    args = parser.parse_args()
    if args.feed_file:
        feed = args.feed_file.read_bytes()
    else:
        query = urllib.parse.urlencode({"search_query": "ti:SceneWeave", "start": 0, "max_results": 20})
        request = urllib.request.Request(
            f"https://export.arxiv.org/api/query?{query}",
            headers={"User-Agent": "SceneWeave-paper-link/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            feed = response.read()
    arxiv_id = find_paper(feed)
    if not arxiv_id:
        print("Exact public arXiv record is not available yet; README is unchanged.")
        return
    if update_readme(args.readme, arxiv_id):
        print(f"README updated to https://arxiv.org/abs/{arxiv_id}")
    else:
        print(f"README already links to https://arxiv.org/abs/{arxiv_id}")


if __name__ == "__main__":
    main()
