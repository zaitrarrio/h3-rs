"""Build the static site: web/site/{index.html, demo/index.html}.

The demo page is also compiled into the `h3` binary (`include_str!`), so
`cargo build` picks up edits; this script only assembles the Pages tree.
Re-run after editing web/*.html; `scripts/deploy-pages.sh` deploys it.
"""
import pathlib, shutil
root = pathlib.Path(__file__).resolve().parent
site = root / "site"
shutil.rmtree(site, ignore_errors=True)
(site / "demo").mkdir(parents=True)
shutil.copy(root / "landing.html", site / "index.html")
shutil.copy(root / "demo.html", site / "demo/index.html")
(site / "_headers").write_text("/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: strict-origin-when-cross-origin\n")
print("wrote", site)
