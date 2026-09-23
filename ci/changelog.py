#!/usr/bin/env python3
"""Stage changelog entries as one file per change; assemble them at release.

Every entry used to be inserted at the same anchor -- the top of
``## [Unreleased]`` -> ``### Fixed`` -- so any two open pull requests edited the
same region of the same file and git stopped on a conflict. That was the default
outcome, not an occasional one: 17 of the 30 merges before issue #668 touched
``CHANGELOG.md``, and resolving the same region by hand repeatedly is where two
shipped defects came from (#662 committed literal conflict markers, #656
duplicated its entry verbatim).

``CHANGELOG.md merge=union`` in ``.gitattributes`` (#669) fixes *local* merges
only: GitHub's merge machinery does not read ``.gitattributes``, measured on
#663 before and after that driver landed. Fragments remove the collision from
the data model instead, so there is nothing for either path to detect.

    changelog.d/659.fixed.md      one file, one entry, one author
    changelog.d/654.fixed.md      no shared anchor, so no conflict

Subcommands::

    python3 ci/changelog.py check              names, shape, size (CI + pre-commit)
    python3 ci/changelog.py required           a source PR carries a fragment
    python3 ci/changelog.py render             preview the assembled section
    python3 ci/changelog.py build --version X  fold fragments into CHANGELOG.md

Why this and not towncrier, which is the same pattern off the shelf:

* The release path installs nothing. ``release.yml``'s ``github-release`` job is
  ``actions/checkout`` plus ``python3 ci/release_notes.py`` -- no ``setup-python``,
  no ``pip install``, no venv. Every script in ``ci/`` is stdlib-only for that
  reason, and towncrier would put a dependency resolution between a tag push and
  a published release.
* The output format is already fixed and is not towncrier-shaped. Keep a
  Changelog headings, in a prescribed order, carrying multi-paragraph bullets
  with nested lists -- pinned by ``test_changelog_structure.py`` on one side and
  parsed by ``ci/release_notes.py`` on the other. Reproducing it needs a custom
  Jinja template plus ``title_format``, ``issue_format`` and wrap settings: more
  configuration surface than the assembler below is code.
* The checks that motivate half of #668 are not towncrier's. ``towncrier check``
  asks only "is a fragment present"; it has nothing for the size budget, and
  nothing for "the PR must not also edit CHANGELOG.md", which is the rule that
  actually removes the conflict.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_DIR = REPO_ROOT / "changelog.d"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

#: Keep a Changelog's subsections, in the order it prescribes. Lowercased here
#: because that is how they appear in a filename; ``test_changelog_structure.py``
#: holds the same list title-cased, and a test asserts the two agree.
CATEGORIES = ("added", "changed", "deprecated", "removed", "fixed", "security")

#: Files that live in ``changelog.d/`` without being fragments.
NON_FRAGMENTS = frozenset({"README.md", ".gitkeep"})

#: ``659.fixed.md``, or ``659.fixed.2.md`` for a second entry of one kind under
#: one issue, or ``+some-slug.fixed.md`` for a change with no issue (towncrier's
#: convention for the same case, kept so the naming is not novel).
FRAGMENT_NAME = re.compile(
    r"^(?:(?P<issue>[0-9]+)|\+(?P<slug>[a-z0-9][a-z0-9-]*))"
    r"\.(?P<category>" + "|".join(CATEGORIES) + r")"
    r"(?:\.(?P<seq>[0-9]+))?\.md$"
)

#: A fragment is one bullet, rendered exactly as it will appear in the file.
#: Verbatim rather than reflowed on purpose: an entry carries nested ``  * ``
#: lists and inline code, and an assembler that re-indented them would be one
#: more thing between what a contributor writes and what ships.
MAX_FRAGMENT_CHARS = 1200

#: Paths whose change is user-visible and therefore needs an entry. The shipped
#: library only: ``python/tests/`` is deliberately absent, so a test-only or
#: docs-only or CI-only pull request is not asked for a changelog entry it has
#: nothing to say in.
SOURCE_PREFIXES = ("python/bngsim/", "src/", "include/")

_UNRELEASED = "## [Unreleased]"


@dataclass(frozen=True)
class Fragment:
    """One staged entry: its file, its sort key, and its rendered bullet."""

    path: Path
    category: str
    issue: int | None
    slug: str | None
    seq: int
    text: str

    @property
    def sort_key(self) -> tuple:
        # Highest issue number first, which is the newest-on-top order the file
        # has always had -- and is now chosen by the assembler rather than by
        # whichever contributor got to the anchor first. Issueless ``+slug``
        # fragments sort after the numbered ones, alphabetically.
        if self.issue is not None:
            return (0, -self.issue, self.seq)
        return (1, self.slug or "", self.seq)


def load(directory: Path = FRAGMENT_DIR) -> list[Fragment]:
    """Every parseable fragment in ``directory``, in assembly order.

    Unparseable names are skipped here and reported by :func:`validate`; loading
    and validating are separate so ``render`` and ``build`` cannot be derailed by
    a file the check would have rejected anyway.
    """
    fragments = []
    for path in sorted(directory.glob("*")) if directory.is_dir() else []:
        if path.name in NON_FRAGMENTS or not path.is_file():
            continue
        m = FRAGMENT_NAME.match(path.name)
        if not m:
            continue
        fragments.append(
            Fragment(
                path=path,
                category=m.group("category"),
                issue=int(m.group("issue")) if m.group("issue") else None,
                slug=m.group("slug"),
                seq=int(m.group("seq") or 1),
                text=path.read_text(encoding="utf-8").strip("\n"),
            )
        )
    return sorted(fragments, key=lambda f: f.sort_key)


def _display(path: Path) -> str:
    """``changelog.d/659.fixed.md`` when the path is in the checkout, else as-is.

    A message naming an absolute temp path is no use to a contributor reading a
    CI log, and ``relative_to`` raises rather than falling back -- which took the
    whole check down when ``validate`` was pointed at a directory outside the
    repository.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def validate(directory: Path = FRAGMENT_DIR) -> list[str]:
    """Everything wrong with the staged fragments, as reader-facing messages."""
    problems: list[str] = []
    if not directory.is_dir():
        return [f"{_display(directory)}/ is missing"]

    seen: dict[str, Path] = {}
    for path in sorted(directory.glob("*")):
        if path.name in NON_FRAGMENTS:
            continue
        rel = _display(path)
        if not path.is_file():
            problems.append(f"{rel}: not a file; changelog.d/ holds fragments, nothing else")
            continue
        m = FRAGMENT_NAME.match(path.name)
        if not m:
            problems.append(
                f"{rel}: name does not parse. Use <issue>.<category>.md — e.g. "
                f"659.fixed.md — with category one of {', '.join(CATEGORIES)}. "
                f"A second entry of one kind for one issue is 659.fixed.2.md; a "
                f"change with no issue is +short-slug.fixed.md."
            )
            continue

        # Two fragments differing only in category are fine (one change can be
        # both an Added and a Fixed); two with the same key are a duplicate.
        # The issue number is normalized, so 659 and 0659 are one issue and
        # 659.fixed.md and 659.fixed.1.md are one entry -- which is how #656
        # would look now that a name collision can no longer produce it.
        who = int(m.group("issue")) if m.group("issue") else m.group("slug")
        key = f"{who}.{m.group('category')}.{int(m.group('seq') or 1)}"
        if key in seen:
            problems.append(f"{rel}: duplicates {seen[key].name}")
        seen[key] = path

        text = path.read_text(encoding="utf-8").strip("\n")
        if not text.strip():
            problems.append(f"{rel}: is empty")
            continue
        if not text.startswith("- "):
            problems.append(
                f"{rel}: must start with '- '. A fragment is the bullet exactly as "
                f"it will read in CHANGELOG.md, so assembly is concatenation and "
                f"nothing reflows your text."
            )
        for lineno, line in enumerate(text.split("\n")[1:], start=2):
            if line and not line.startswith("  "):
                problems.append(
                    f"{rel}:{lineno}: continuation lines belong to the bullet and must "
                    f"be indented by two spaces (nested bullets are '  * ')."
                )
                break
        if len(text) > MAX_FRAGMENT_CHARS:
            problems.append(
                f"{rel}: {len(text):,} characters, over the {MAX_FRAGMENT_CHARS:,} limit. "
                f"An entry says what changed and what it affected; the reasoning "
                f"belongs in issue #{m.group('issue') or '…'} and the commit message, "
                f"which is where a reader who wants it will look."
            )
    return problems


def render(fragments: list[Fragment]) -> str:
    """The ``### Kind`` subsections the staged fragments assemble into."""
    out: list[str] = []
    for category in CATEGORIES:
        entries = [f for f in fragments if f.category == category]
        if not entries:
            continue
        out.append(f"### {category.title()}")
        out.append("")
        for entry in entries:
            out.append(entry.text)
            out.append("")
    return "\n".join(out).strip("\n")


def _split_unreleased(text: str) -> tuple[list[str], list[str], list[str]]:
    """``CHANGELOG.md`` as (head through the Unreleased heading, its body, tail)."""
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith(_UNRELEASED)]
    if len(starts) != 1:
        raise SystemExit(f"changelog: expected one '{_UNRELEASED}' heading, found {len(starts)}")
    start = starts[0]
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## [")), len(lines)
    )
    return lines[: start + 1], lines[start + 1 : end], lines[end:]


def _sections(body: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Split an ``[Unreleased]`` body into its preamble and ``### Kind`` blocks.

    Heading-level only. Entry boundaries inside a section are never parsed: an
    entry can contain a blank line, a nested list or a fenced block, and a parser
    that guessed where one ended would be one more way to lose prose. The section
    text is carried through verbatim.
    """
    preamble: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in body:
        heading = re.match(r"^### +(?P<kind>.+?)\s*$", line)
        if heading:
            current = heading.group("kind").strip().lower()
            sections.setdefault(current, [])
            continue
        (sections[current] if current else preamble).append(line)
    return preamble, sections


def build(text: str, fragments: list[Fragment], version: str, when: str) -> str:
    """``CHANGELOG.md`` with a new ``## [version]`` section assembled below a
    fresh, empty ``## [Unreleased]``.

    Anything already sitting under ``[Unreleased]`` by hand is folded in rather
    than replaced. That is what makes the migration safe -- five unreleased
    entries predate fragments -- and it stays afterwards as the guarantee that
    this command cannot silently drop prose someone wrote.
    """
    head, body, tail = _split_unreleased(text)
    preamble, existing = _sections(body)

    blocks: list[str] = [f"## [{version}] - {when}"]
    for category in CATEGORIES:
        staged = [f.text for f in fragments if f.category == category]
        carried = "\n".join(existing.pop(category, [])).strip("\n")
        if not staged and not carried:
            continue
        blocks.append(f"### {category.title()}")
        blocks.extend(staged)
        if carried:
            blocks.append(carried)
    # A heading outside the Keep a Changelog vocabulary is kept rather than
    # dropped. test_changelog_structure.py forbids one, so this should be
    # unreachable -- but losing prose to a heading typo is not a trade to make.
    for kind, lines in existing.items():
        carried = "\n".join(lines).strip("\n")
        if carried:
            blocks += [f"### {kind.title()}", carried]

    unreleased = [_UNRELEASED, "\n".join(preamble).strip("\n")]
    assembled = "\n\n".join(
        part
        for part in [
            "\n".join(head[:-1]).strip("\n"),
            *(p for p in unreleased if p),
            "\n\n".join(blocks),
            "\n".join(tail).strip("\n"),
        ]
        if part
    )
    return assembled.rstrip("\n") + "\n"


def required(changed: list[str]) -> list[str]:
    """What is wrong with a pull request's changed-file list, if anything.

    Two rules, and the second is the one that actually removes the conflict: a
    fragment does no good if the branch also edits the shared file.
    """
    problems: list[str] = []
    source = sorted(p for p in changed if p.startswith(SOURCE_PREFIXES))
    staged = [
        p for p in changed if p.startswith("changelog.d/") and FRAGMENT_NAME.match(Path(p).name)
    ]

    if source and not staged:
        problems.append(
            "This branch changes the shipped library ("
            + ", ".join(source[:3])
            + (f", +{len(source) - 3} more" if len(source) > 3 else "")
            + ") and stages no changelog entry. Add one file under changelog.d/ — "
            "see changelog.d/README.md — or label the pull request 'changelog exempt' "
            "if the change is genuinely invisible to users."
        )
    if "CHANGELOG.md" in changed:
        problems.append(
            "This branch edits CHANGELOG.md. That file has one anchor, so every "
            "branch that edits it conflicts with every other one (issue #668); it "
            "is now written only by the release commit. Move the entry to "
            "changelog.d/<issue>.<category>.md."
        )
    return problems


def _changed_files(base: str) -> list[str]:
    """Paths this branch changes relative to ``base``, as git reports them."""
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in diff.stdout.split("\n") if line]


def would_truncate(assembled: str, version: str) -> tuple[bool, int, int]:
    """Whether ``release.yml`` would publish this version's notes truncated.

    The size gate, placed where it can still be acted on. ``ci/release_notes.py``
    cuts a section over GitHub's limit at a paragraph boundary and appends a
    footer pointing back at ``CHANGELOG.md`` -- graceful, and how 0.13.0 and
    0.12.0 both published incomplete notes without anyone noticing. Asking the
    same question at assembly time puts it in front of the one person who can
    still shorten an entry.

    The limit and the truncation are read from ``release_notes`` rather than
    re-derived, so there is one 125,000 in the repository and this cannot drift
    away from what the workflow actually does.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import release_notes

    notes = release_notes.build(assembled, version)
    return "are truncated here" in notes, len(notes), release_notes.LIMIT


def _report(problems: list[str], ok: str) -> int:
    if not problems:
        print(ok)
        return 0
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="validate the staged fragments")

    req = sub.add_parser("required", help="a source change must stage a fragment")
    req.add_argument("--base", default="origin/main", help="branch point to diff against")
    req.add_argument(
        "--files",
        help="read the changed-file list from this file ('-' for stdin) instead of git",
    )

    sub.add_parser("render", help="print the section the fragments assemble into")

    bld = sub.add_parser("build", help="fold the fragments into CHANGELOG.md")
    bld.add_argument("--version", required=True)
    bld.add_argument("--date", default=_date.today().isoformat())
    bld.add_argument(
        "--allow-truncation",
        action="store_true",
        help="assemble even if the GitHub Release body would be truncated",
    )

    args = parser.parse_args(argv)

    if args.command == "check":
        return _report(validate(), f"changelog: {len(load())} fragment(s) staged, all well-formed")

    if args.command == "required":
        if args.files:
            raw = sys.stdin.read() if args.files == "-" else Path(args.files).read_text()
            changed = [line.strip() for line in raw.split("\n") if line.strip()]
        else:
            changed = _changed_files(args.base)
        # Fail closed. The workflow pipes `gh api … --jq` into this, and a
        # pipeline's exit status is the last command's: without this, an API
        # call that failed for any reason would hand over an empty list, every
        # rule would find nothing to object to, and the gate would go green
        # having read nothing. That is #664's defect -- a check that returns
        # success without opening a file -- and it is the one failure a gate
        # cannot afford. No pull request changes zero files.
        if not changed:
            return _report(
                ["no changed files were reported, so nothing was actually checked"],
                "",
            )
        return _report(required(changed), "changelog: this branch's entry is staged correctly")

    fragments = load()
    if args.command == "render":
        print(render(fragments))
        return 0

    problems = validate()
    if problems:
        return _report(problems, "")
    if not fragments:
        print("changelog: no fragments staged", file=sys.stderr)

    text = CHANGELOG.read_text(encoding="utf-8")
    assembled = build(text, fragments, args.version, args.date)
    truncated, size, limit = would_truncate(assembled, args.version)
    if truncated and not args.allow_truncation:
        print(
            f"error: {args.version}'s notes exceed GitHub's {limit:,}-character "
            f"release-body limit and would publish truncated. Shorten the longest "
            f"entries, or pass --allow-truncation to accept the footer that points "
            f"at CHANGELOG.md.",
            file=sys.stderr,
        )
        return 1

    CHANGELOG.write_text(assembled, encoding="utf-8")
    for fragment in fragments:
        fragment.path.unlink()
    print(
        f"changelog: {len(fragments)} fragment(s) -> ## [{args.version}] - {args.date}; "
        f"release body is {size:,} of {limit:,} characters."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
