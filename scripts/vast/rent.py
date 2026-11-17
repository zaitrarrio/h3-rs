"""Rent a Vast.ai offer with the h3 bootstrap as its onstart line.

    python3 rent.py OFFER_ID "ONSTART_LINE" [DISK_GB]

Prints the new contract id and the instance API key (for the box-side
destroy timer: `curl -X DELETE -H 'Authorization: Bearer <key>'
https://console.vast.ai/api/v0/instances/<id>/`). Filter offers on
`compute_cap >= 800` first (`POST /api/v0/bundles/`); instances are listed
at `GET /api/v1/instances/`, destroyed at `DELETE /api/v0/instances/<id>/`.
"""
import json, sys
import vast_api as v

offer, onstart = int(sys.argv[1]), sys.argv[2]
disk = int(sys.argv[3]) if len(sys.argv) > 3 else 250
body = {
    "client_id": "me",
    "image": "nvidia/cuda:12.8.1-runtime-ubuntu24.04",
    "disk": disk,
    "runtype": "ssh",
    "onstart": onstart,
    "env": {"NVIDIA_DRIVER_CAPABILITIES": "all"},
    "label": "h3-rs",
}
r = v.call("PUT", f"/asks/{offer}/", body=body)
print(json.dumps(r))
