"""RunPod REST (rest.runpod.io/v1) from $RUNPOD_API_KEY.

    rp.py volumes                      network volumes (id, name, size, data center)
    rp.py gpus [FILTER]                GPU types with secure-cloud price and stock
    rp.py dcs GPU_ID                   data centers listing GPU_ID as available
    rp.py create NAME GPU_ID DC VOLUME_ID "START_LINE" [DISK_GB]
                                       secure-cloud pod on the volume (mounted at
                                       /workspace), image nvidia/cuda:12.8.1-runtime-
                                       ubuntu24.04, START_LINE as `bash -c`, the API
                                       key in its env for self-termination. DC `-`
                                       lets RunPod pick the data center, VOLUME_ID `-`
                                       mounts no volume (a kernel-check pod)
    rp.py get POD | list | stop POD | start POD | terminate POD
"""
import json, os, sys, urllib.request
KEY = os.environ["RUNPOD_API_KEY"]
IMAGE = "nvidia/cuda:12.8.1-runtime-ubuntu24.04"

def rest(method, path, body=None):
    req = urllib.request.Request("https://rest.runpod.io/v1" + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "h3-rs-runpod/1"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path}: HTTP {e.code}: {e.read().decode(errors='replace')[:600]}")

def gql(q):
    req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": q}).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

op = sys.argv[1]
if op == "volumes":
    for v in rest("GET", "/networkvolumes"):
        print(v["id"], v["dataCenterId"], f'{v["size"]} GB', v["name"])
elif op == "gpus":
    flt = sys.argv[2] if len(sys.argv) > 2 else ""
    for t in gql('{ gpuTypes { id memoryInGb secureCloud lowestPrice(input:{gpuCount:1, secureCloud:true}) { uninterruptablePrice stockStatus } } }')["data"]["gpuTypes"]:
        if flt.lower() in t["id"].lower():
            print(t["id"], f'{t["memoryInGb"]} GB', t["lowestPrice"])
elif op == "dcs":
    for dc in gql("{ dataCenters { id listed gpuAvailability { gpuTypeId available stockStatus } } }")["data"]["dataCenters"]:
        for a in dc["gpuAvailability"] or []:
            if a["gpuTypeId"] == sys.argv[2] and a["available"]:
                print(dc["id"], a["stockStatus"])
elif op == "create":
    name, gpu, dc, vol, start = sys.argv[2:7]
    disk = int(sys.argv[7]) if len(sys.argv) > 7 else 50
    body = {
        "name": name, "imageName": IMAGE, "cloudType": "SECURE", "computeType": "GPU",
        "gpuTypeIds": [gpu], "gpuCount": 1,
        "containerDiskInGb": disk, "volumeInGb": 0, "ports": [],
        "dockerStartCmd": ["bash", "-c", start],
        "env": {"RUNPOD_API_KEY": KEY, "NVIDIA_DRIVER_CAPABILITIES": "all"},
    }
    if dc != "-":
        body["dataCenterIds"] = [dc]
    if vol != "-":
        body.update(networkVolumeId=vol, volumeMountPath="/workspace")
    r = rest("POST", "/pods", body)
    print(json.dumps({k: r.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "machine", "networkVolume")}))
elif op == "get":
    r = rest("GET", f"/pods/{sys.argv[2]}")
    print(json.dumps({k: r.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "lastStatusChange", "machine", "publicIp")}))
elif op == "list":
    for p in rest("GET", "/pods"):
        print(p["id"], p["desiredStatus"], p.get("costPerHr"), p["name"])
elif op == "stop":
    print(rest("POST", f"/pods/{sys.argv[2]}/stop"))
elif op == "start":
    print(rest("POST", f"/pods/{sys.argv[2]}/start"))
elif op == "terminate":
    print(rest("DELETE", f"/pods/{sys.argv[2]}"))
else:
    sys.exit(__doc__)
