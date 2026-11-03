#!/usr/bin/env python3
"""Verify aggregate sources and regenerate all five real-data Clustro figures.

No clustering refit. No patient-level export. Produces vector PDFs, 600-DPI PNGs,
and an aggregate-only SHA-256 manifest. Requires the existing local study outputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from regenerate_figures_4_5_checked import ROOT, check, required, sha256


FILES = ["fig1_workflow", "fig2_model_selection", "fig3_null_reference",
         "fig4_phenotypes", "fig5_robustness"]


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reporting-dir", default=str(ROOT / "results/methods_revision_reporting"))
    parser.add_argument("--outdir", default=str(ROOT / "results/final_paper_figures_v8610"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-frozen-hash-check", action="store_true",
                        help="Only for inspecting aggregate outputs, not for final figures")
    parser.add_argument("--dpi", type=int, default=600)
    args=parser.parse_args()
    if args.dpi < 500:
        parser.error("Publication PNGs must be generated at >=500 DPI")
    report_dir=Path(args.reporting_dir).resolve()
    outdir=Path(args.outdir).resolve()
    try:
        result=check(report_dir, not args.skip_frozen_hash_check)
        if args.verify_only:
            print(json.dumps(result["cohorts"],indent=2))
            print("PRE-FLIGHT PASSED. No figures written.")
            return 0
        outdir.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ, CLUSTRO_REPORTING_DIR=str(report_dir))
        subprocess.run([sys.executable,str(ROOT/"paper_figures/make_figures.py"),
                        "--outdir",str(outdir),"--pdf","--dpi",str(args.dpi)],
                       cwd=ROOT,env=env,check=True)
        result["png_dpi_requested"]=args.dpi
        result["figure_sha256"]={}
        for stem in FILES:
            for ext in ("png","pdf"):
                path=required(outdir/(stem+"."+ext))
                result["figure_sha256"][path.name]=sha256(path)
        # PNG dimension sanity check without changing any figure.
        from PIL import Image
        result["png_dimensions_px"]={}
        for stem in FILES:
            with Image.open(outdir/(stem+".png")) as im:
                w,h=im.size
                if w < 4000:
                    raise RuntimeError(f"{stem} width {w}px is unexpectedly small")
                result["png_dimensions_px"][stem]=[w,h]
        (outdir/"figure_source_manifest_v8610.json").write_text(
            json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print("PASS: source checks, figure output dimensions and SHA-256 manifest")
        print(outdir)
        return 0
    except (FileNotFoundError,ValueError,KeyError,RuntimeError,subprocess.CalledProcessError) as exc:
        print("FAILED: "+str(exc),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
