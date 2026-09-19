"""
Regenerate every figure in the Clustro set.

    python make_figures.py                 # writes PNGs next to this file
    python make_figures.py --outdir out    # writes into ./out
    python make_figures.py --pdf           # also writes vector PDFs
    python make_figures.py --only 3 5      # just figures 3 and 5
"""
from __future__ import annotations

import argparse
import os
import sys

import fig1_workflow
import fig2_model_selection
import fig3_null_reference
import fig4_phenotypes
import fig5_robustness

MODULES = {
    1: fig1_workflow,
    2: fig2_model_selection,
    3: fig3_null_reference,
    4: fig4_phenotypes,
    5: fig5_robustness,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--only", nargs="*", type=int, choices=sorted(MODULES))
    ap.add_argument("--pdf", action="store_true",
                    help="also save a vector PDF of each figure")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    wanted = args.only or sorted(MODULES)

    for n in wanted:
        mod = MODULES[n]
        path = mod.build(args.outdir)
        print(f"figure {n}: {path}")
        if args.pdf:
            import clustro_style as cs
            orig_save = cs.save

            def pdf_save(fig, p, dpi=args.dpi, _o=orig_save):
                return _o(fig, os.path.splitext(p)[0] + ".pdf", dpi=dpi)

            cs.save = pdf_save
            mod.save = pdf_save
            print(f"figure {n}: {mod.build(args.outdir)}")
            cs.save = orig_save
            mod.save = orig_save


if __name__ == "__main__":
    sys.exit(main())
