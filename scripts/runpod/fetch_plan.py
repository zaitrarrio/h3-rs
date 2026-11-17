"""Print fetch lines `URL SIZE DEST` for weight trees, for a pod command.

    python3 fetch_plan.py r2:weights/h3-8step/<sha>/=h3-8step:transformer/,fastvideo_inference.json \
                          hf:Pika-Labs/Pika-H3-Flex=pika-h3-flex:fl2va/,ref2va/,processor/ ...

r2:PREFIX=DEST[:INCLUDE,...] lists the fv-weights bucket under PREFIX (6-hour
presigned GETs); hf:REPO=DEST[:INCLUDE,...] lists the Hugging Face repo. An
INCLUDE is a directory (trailing /) or an exact path; none means everything.
DEST is relative to /workspace/weights. The pod runs each line as
`[ -f DEST ] && [ $(stat -c %s DEST) = SIZE ] || curl -o DEST URL`, so a run
that stops can be queued again and finishes what is missing.
"""
import json, os, sys, urllib.request
exec(open(os.path.join(os.path.dirname(__file__), "..", "vast", "r2.py")).read())
get = lambda b, k: s3.generate_presigned_url("get_object", Params={"Bucket": b, "Key": k}, ExpiresIn=6 * 3600)
def keep(rel, inc):
    return not inc or any(rel == i or (i.endswith("/") and rel.startswith(i)) for i in inc)
for spec in sys.argv[1:]:
    kind, rest = spec.split(":", 1)
    src, dest = rest.split("=", 1)
    inc = []
    if ":" in dest:
        dest, incs = dest.split(":", 1)
        inc = [i for i in incs.split(",") if i]
    if kind == "r2":
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket="fv-weights", Prefix=src):
            for o in page.get("Contents", []):
                rel = o["Key"][len(src):]
                if rel and keep(rel, inc):
                    print(get("fv-weights", o["Key"]), o["Size"], f"/workspace/weights/{dest}/{rel}")
    elif kind == "hf":
        tree = json.loads(urllib.request.urlopen(f"https://huggingface.co/api/models/{src}/tree/main?recursive=true", timeout=60).read())
        for e in tree:
            if e["type"] == "file" and keep(e["path"], inc):
                print(f"https://huggingface.co/{src}/resolve/main/{e['path']}", e["size"], f"/workspace/weights/{dest}/{e['path']}")
    else:
        sys.exit(f"unknown kind {kind}")
