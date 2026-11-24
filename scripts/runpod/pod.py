"""Create a RunPod pod through GraphQL (api.runpod.io), retrying while the
data centre has no stock. rest.runpod.io is unreachable from the agent
sandbox, so this is the path `rp.py create` cannot take there.

    pod.py GPU_TYPE DC VOLUME NAME START_FILE [DISK_GB] [TRIES]
      GPU_TYPE  e.g. "NVIDIA RTX PRO 6000 Blackwell Server Edition"
      DC        e.g. EU-RO-1
      VOLUME    network volume id, or - for none (mounted at /workspace)
      START_FILE  the boot command (make_boot.py's output)
      DISK_GB   container disk (default 300); TRIES (default 90, one a minute)

Prints `POD <id> <$/h> <gpu>` and exits 0, or exits 1 after TRIES.

The pod gets no account key: RunPod injects a pod-scoped RUNPOD_API_KEY,
which is what terminates the pod through GraphQL (the boot's `terminate`);
an account key in the pod's env would replace it, and RunPod rejects account
keys from inside pods (401 `{"error":{}}`).
"""
import json
import os
import shlex
import sys
import time
import urllib.request

KEY = os.environ["RUNPOD_API_KEY"]
gpu, dc, vol, name, start_file = sys.argv[1:6]
disk = int(sys.argv[6]) if len(sys.argv) > 6 else 300
tries = int(sys.argv[7]) if len(sys.argv) > 7 else 90
start = open(start_file).read().strip()
s = json.dumps
volume = f'networkVolumeId: {s(vol)}, volumeMountPath: "/workspace",' if vol != "-" else ""
q = f"""mutation {{ podFindAndDeployOnDemand(input: {{
  cloudType: SECURE, gpuCount: 1, gpuTypeId: {s(gpu)}, dataCenterId: {s(dc)},
  name: {s(name)}, imageName: "nvidia/cuda:12.8.1-runtime-ubuntu24.04",
  dockerArgs: {s("bash -c " + shlex.quote(start))}, containerDiskInGb: {disk}, volumeInGb: 0, ports: "",
  {volume}
  env: [{{key: "NVIDIA_DRIVER_CAPABILITIES", value: "all"}}]
}}) {{ id costPerHr machine {{ gpuDisplayName }} }} }}"""
for i in range(tries):
    try:
        req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": q}).encode(),
                                     headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                              "User-Agent": "h3-rs-pod/1"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    except Exception as e:  # network hiccup: retry
        print("try", i, type(e).__name__, flush=True)
        time.sleep(60)
        continue
    if r.get("errors"):
        print("try", i, r["errors"][0].get("extensions", {}).get("code") or r["errors"][0].get("message"), flush=True)
        time.sleep(60)
        continue
    p = r["data"]["podFindAndDeployOnDemand"]
    print("POD", p["id"], p.get("costPerHr"), p["machine"]["gpuDisplayName"], flush=True)
    sys.exit(0)
sys.exit(1)
