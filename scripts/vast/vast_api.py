import json, os, sys, urllib.request, urllib.parse
KEY = os.environ["VAST_API_KEY"]
BASE = "https://console.vast.ai/api/v0"
def call(method, path, body=None, query=None):
    url = BASE + path
    if query is not None:
        url += "?q=" + urllib.parse.quote(json.dumps(query))
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"{}")
