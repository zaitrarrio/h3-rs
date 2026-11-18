"""Run the h3 container image (Dockerfile, built by .github/workflows/container.yml)
on RunPod: as a service on a pod, and as a CLI on a serverless endpoint.

Talks to RunPod's GraphQL API (api.runpod.io) and its serverless API
(api.runpod.ai) with $RUNPOD_API_KEY; the image is private, so RunPod pulls it
with a registry credential (`--auth NAME`, added on RunPod: Settings, Container
Registry Auth, a GitHub token with read:packages only).

    container.py auths                                  registry credential names
    container.py pod IMAGE AUTH GPU DC VOLUME -- ARGS…  a pod running `h3 ARGS…`, port 7860 over RunPod's proxy
    container.py pod-status POD_ID | pod-terminate POD_ID
    container.py endpoint IMAGE AUTH GPU_POOL VOLUME DC  a serverless endpoint running `h3 worker` (0..1 workers)
    container.py run ENDPOINT_ID job.json                submit a job ({"input": {"args": […], "uploads": {…}}}), wait, print the result
    container.py endpoint-delete ENDPOINT_ID

The pod serves https://POD_ID-7860.proxy.runpod.net/. A network volume mounts
at /workspace on a pod and at /runpod-volume on a serverless worker. The
fv-weights volume is shared: never a pod and a worker on it at once.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

KEY = os.environ["RUNPOD_API_KEY"]
UA = "h3-rs-container/1"


def call(url, body=None, method=None):
    req = urllib.request.Request(url, method=method or ("POST" if body is not None else "GET"),
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                          "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"{url}: HTTP {e.code}: {e.read().decode(errors='replace')[:800]}")


def gql(q):
    r = call("https://api.runpod.io/graphql", {"query": q})
    if r.get("errors"):
        sys.exit(f"GraphQL: {json.dumps(r['errors'])[:800]}")
    return r["data"]


def s(v):
    return json.dumps(v)


def auth_id(name):
    for c in gql("{ myself { containerRegistryCreds { id name } } }")["myself"]["containerRegistryCreds"]:
        if c["name"] == name:
            return c["id"]
    sys.exit(f"no registry credential named {name!r} (container.py auths)")


op, a = sys.argv[1], sys.argv[2:]
if op == "auths":
    for c in gql("{ myself { containerRegistryCreds { id name } } }")["myself"]["containerRegistryCreds"]:
        print(c["name"])
elif op == "pod":
    image, auth, gpu, dc, volume = a[:5]
    args = a[a.index("--") + 1:]
    q = f"""mutation {{ podFindAndDeployOnDemand(input: {{
        cloudType: SECURE, gpuCount: 1, gpuTypeId: {s(gpu)}, dataCenterId: {s(dc)},
        name: "h3-rs-container-service", imageName: {s(image)}, containerRegistryAuthId: {s(auth_id(auth))},
        dockerArgs: {s(" ".join(args))}, ports: "7860/http", containerDiskInGb: 20, volumeInGb: 0,
        networkVolumeId: {s(volume)}, volumeMountPath: "/workspace", env: [{{key: "RUST_BACKTRACE", value: "1"}}]
    }}) {{ id imageName machine {{ gpuDisplayName }} }} }}"""
    p = gql(q)["podFindAndDeployOnDemand"]
    print(p["id"], p["imageName"], p["machine"])
    print(f"https://{p['id']}-7860.proxy.runpod.net/")
elif op == "pod-status":
    p = gql(f"""{{ pod(input: {{podId: {s(a[0])}}}) {{ id desiredStatus lastStatusChange
        runtime {{ uptimeInSeconds gpus {{ id gpuUtilPercent memoryUtilPercent }} }} }} }}""")["pod"]
    print(json.dumps(p))
elif op == "pod-terminate":
    gql(f"mutation {{ podTerminate(input: {{podId: {s(a[0])}}}) }}")
    print("terminated", a[0])
elif op == "endpoint":
    image, auth, pool, volume, dc = a[:5]
    name = f"h3-rs-cli-{int(time.time())}"
    t = gql(f"""mutation {{ saveTemplate(input: {{ name: {s(name)}, imageName: {s(image)}, isServerless: true,
        containerRegistryAuthId: {s(auth_id(auth))}, dockerArgs: "worker", containerDiskInGb: 20, volumeInGb: 0,
        env: [{{key: "RUST_BACKTRACE", value: "1"}}] }}) {{ id }} }}""")["saveTemplate"]
    e = gql(f"""mutation {{ saveEndpoint(input: {{ name: {s(name)}, templateId: {s(t['id'])}, gpuIds: {s(pool)},
        networkVolumeId: {s(volume)}, locations: {s(dc)}, workersMin: 0, workersMax: 1, idleTimeout: 5,
        scalerType: "QUEUE_DELAY", scalerValue: 4 }}) {{ id }} }}""")["saveEndpoint"]
    print(e["id"], "template", t["id"], name)
elif op == "run":
    endpoint, path = a[:2]
    job = json.load(open(path))
    r = call(f"https://api.runpod.ai/v2/{endpoint}/run", job)
    jid, t0 = r["id"], time.time()
    print("job", jid, r.get("status"), flush=True)
    last = None
    while True:
        st = call(f"https://api.runpod.ai/v2/{endpoint}/status/{jid}")
        if st.get("status") != last:
            last = st.get("status")
            print(f"{time.time() - t0:7.1f}s {last}", flush=True)
        if last in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            print(json.dumps(st, indent=1))
            break
        time.sleep(5)
elif op == "endpoint-delete":
    e = a[0]
    gql(f'mutation {{ updateEndpointWorkersMinMax(input: {{endpointId: {s(e)}, workersMin: 0, workersMax: 0}}) {{ id }} }}')
    time.sleep(5)
    gql(f"mutation {{ deleteEndpoint(id: {s(e)}) }}")
    print("deleted", e)
else:
    sys.exit(__doc__)
