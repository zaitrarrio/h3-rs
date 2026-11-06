"""Drive the Prompt Studio page against a running `h3 serve` (headless
Chromium via Playwright) and save screenshots: studio-1.png, studio-2.png,
studio-loading.png (paths from argv). Used on rentals by scripts/vast.
"""
import sys, time
from playwright.sync_api import sync_playwright
shots = sys.argv[1:4]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 1000})
    pg.on("pageerror", lambda e: print("pageerror:", e))
    pg.goto("http://127.0.0.1:7860/")
    pg.wait_for_function("!document.getElementById('go').disabled", timeout=120000)
    print("engine:", pg.text_content("#eng-text"))
    runs = [("A golden retriever puppy bounds through a pile of autumn leaves in a park, slow motion, warm sunset light; leaves rustle and the puppy pants happily.", "1344x768", "124", "11"),
            ("A crowded night market in the rain, neon signs reflected in puddles, steam rising from food stalls, chatter and sizzling woks.", "832x480", "124", "5")]
    for i, (prompt, size, length, seed) in enumerate(runs):
        pg.fill("#prompt", prompt); pg.select_option("#size", size); pg.select_option("#length", length); pg.fill("#seed", seed)
        pg.evaluate("document.getElementById('summary').textContent = ''")
        t0 = time.time(); pg.click("#go")
        if i == 0:
            pg.wait_for_function("document.getElementById('summary').textContent === '' && !document.getElementById('empty').hidden || document.querySelector('.stage.active')", timeout=60000)
            pg.screenshot(path="/w/studio-loading.png")
        pg.wait_for_function("document.getElementById('summary').textContent.includes('total')", timeout=1800000)
        print(f"run {i}: {time.time() - t0:.1f}s wall |", pg.text_content("#summary"))
        print("   stages:", " | ".join(s.inner_text().replace("\n", " ") for s in pg.query_selector_all(".stage")))
        print("   frames:", pg.text_content("#counter"), "| history:", len(pg.query_selector_all(".clip")))
        pg.wait_for_timeout(1500)
        pg.screenshot(path=f"/w/studio-{i + 1}.png", full_page=True)
    b.close()
