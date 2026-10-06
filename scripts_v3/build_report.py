"""Phase 4.1b: assemble results_v3/IAMA_Net_report_v3.md from the narrative
template (results_v3/report/report_template.md) by substituting
{TABLE:name} with script-generated table fragments and {FIG:name|caption}
with figure embeds. Every numeric table is generated from CSVs by
make_report_tables.py — no numbers are hand-typed in the narrative.
"""

import re
import sys
from pathlib import Path

from iama_env import PROJECT_ROOT

REP = PROJECT_ROOT / "results_v3" / "report"
OUT = PROJECT_ROOT / "results_v3" / "IAMA_Net_report_v3.md"


def main():
    tpl = (REP / "report_template.md").read_text(encoding="utf-8")

    def table(m):
        name = m.group(1)
        p = REP / "tables" / name
        if not p.exists():
            return f"> **[missing table: {name} — regenerate with make_report_tables.py]**"
        return p.read_text(encoding="utf-8").strip()

    def fig(m):
        name, caption = m.group(1), m.group(2)
        return f"![{caption}](report/figures/{name})\n\n*Figure: {caption}*"

    out = re.sub(r"\{TABLE:([\w.\-]+)\}", table, tpl)
    out = re.sub(r"\{FIG:([\w.\-]+)\|([^}]+)\}", fig, out)
    OUT.write_text(out, encoding="utf-8")
    print(f"wrote {OUT} ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
