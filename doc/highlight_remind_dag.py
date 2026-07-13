# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""Turn a `snakemake --dag` Graphviz file into the highlighted, legended PNG used in the README.

Usage:
    snakemake -s Snakefile_REMIND <target> --configfile <config> --dag > dag.dot
    python doc/highlight_remind_dag.py dag.dot doc/img/dag_remind_2050.png
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
REMIND_RULES_FILE = REPO_ROOT / "rules" / "REMIND_coupling.smk"

REMIND_STYLE = 'fillcolor="#ffd9a3", color="#b35c00", penwidth=3, style="rounded,filled"'
DEFAULT_STYLE = 'fillcolor="#f2f2f2", color="#999999", penwidth=1, style="rounded,filled"'
REMIND_FONTCOLOR = 'fontcolor="#5c2e00"'
DEFAULT_FONTCOLOR = 'fontcolor="#666666"'

NODE_RE = re.compile(r'^\s*(\d+)\[label = "((?:[^"\\]|\\.)*)",.*\];\s*$')
# Wildcard lines that are constant across this static example and just add noise.
NOISE_SEGMENT_RE = re.compile(r"\\n(scen|iter|year)_REMIND: [^\\]*")

LEGEND_DOT = f"""
digraph legend {{
    graph[bgcolor=white, margin=0.05];
    node[shape=box, style=rounded, fontname=sans, fontsize=10, penwidth=2];
    legend_remind[label = "REMIND-specific rule", {REMIND_STYLE}, {REMIND_FONTCOLOR}];
    legend_default[label = "Standard PyPSA-Eur rule", {DEFAULT_STYLE}, {DEFAULT_FONTCOLOR}];
    legend_remind -> legend_default [style=invis];
}}
"""


def load_remind_rule_names() -> set[str]:
    text = REMIND_RULES_FILE.read_text()
    return set(re.findall(r"^rule (\w+):", text, flags=re.MULTILINE))


def clean_label(label: str) -> str:
    """Drop wildcard lines that are constant for this static example (scenario id, iteration, year)."""
    return NOISE_SEGMENT_RE.sub("", label)


def recolor(lines: list[str], remind_rules: set[str]) -> list[str]:
    remind_ids = set()
    out = []
    for line in lines:
        m = NODE_RE.match(line)
        if not m:
            out.append(line)
            continue
        node_id, label = m.group(1), clean_label(m.group(2))
        rule_name = label.split("\\n", 1)[0]
        is_remind = rule_name in remind_rules
        if is_remind:
            remind_ids.add(node_id)
        style = REMIND_STYLE if is_remind else DEFAULT_STYLE
        fontcolor = REMIND_FONTCOLOR if is_remind else DEFAULT_FONTCOLOR
        out.append(f'\t{node_id}[label = "{label}", {style}, {fontcolor}];\n')

    edge_re = re.compile(r"^\s*(\d+)\s*->\s*(\d+)\s*$")
    final = []
    for line in out:
        m = edge_re.match(line)
        if m and m.group(1) in remind_ids and m.group(2) in remind_ids:
            final.append(f'\t{m.group(1)} -> {m.group(2)} [color="#b35c00", penwidth=2.5];\n')
        else:
            final.append(line)
    return final


def render_png(dot_text: str, png_path: Path) -> None:
    subprocess.run(["dot", "-Tpng", "-o", str(png_path)], input=dot_text, text=True, check=True)


def compose_legend(main_png: Path, out_png: Path, margin: int = 20) -> None:
    """Paste a separately-rendered legend into the top-left corner of the main DAG image.

    Graphviz has no reliable way to pin a disconnected subgraph to an exact corner of a
    hierarchical layout, so the legend is rendered standalone and composited with PIL instead.
    """
    with tempfile.TemporaryDirectory() as tmp:
        legend_png = Path(tmp) / "legend.png"
        render_png(LEGEND_DOT, legend_png)
        main = Image.open(main_png).convert("RGB")
        legend = Image.open(legend_png).convert("RGB")
        main.paste(legend, (margin, margin))
        main.save(out_png)


def main() -> None:
    infile, outfile = sys.argv[1], sys.argv[2]
    remind_rules = load_remind_rule_names()
    lines = Path(infile).read_text().splitlines(keepends=True)
    result = recolor(lines, remind_rules)

    with tempfile.TemporaryDirectory() as tmp:
        main_png = Path(tmp) / "main.png"
        render_png("".join(result), main_png)
        compose_legend(main_png, Path(outfile))

    n_highlighted = sum(1 for line in result if REMIND_STYLE in line)
    print(f"REMIND-specific nodes highlighted: {n_highlighted}")


if __name__ == "__main__":
    main()
