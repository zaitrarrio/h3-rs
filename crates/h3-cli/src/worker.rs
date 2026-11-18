//! `h3 worker`: the `h3` CLI as a RunPod serverless worker, with no Python
//! in the image.
//!
//! The protocol is the one RunPod's own SDK speaks (runpod-python,
//! `serverless/modules/rp_job.py`, `rp_http.py`, `rp_ping.py`):
//!
//! ```text
//! GET  $RUNPOD_WEBHOOK_GET_JOB ($ID → worker id) &job_in_progress=0|1
//!        204 / 400: no job;  200: {"id", "input"} (or a list of them)
//! POST $RUNPOD_WEBHOOK_POST_OUTPUT ($RUNPOD_POD_ID → worker id, $ID → job id) &isStream=false
//!        body: {"output": …} and/or {"error": "…"}
//! GET  $RUNPOD_WEBHOOK_PING ($RUNPOD_POD_ID → worker id) ?job_id=…&runpod_version=…
//!        every $RUNPOD_PING_INTERVAL ms (default 10000)
//! ```
//!
//! with `Authorization: $RUNPOD_AI_API_KEY` on every call. A job's input is
//!
//! ```json
//! {"args": ["generate", "--variant", "draft", "…", "--out", "clip.y4m"],
//!  "uploads": {"clip.y4m": "https://…presigned PUT…", "clip.wav": "…"},
//!  "timeout_s": 1800}
//! ```
//!
//! `args` run as `h3 <args>` (no shell) in a fresh directory per job, so
//! relative output paths land there; `uploads` PUTs files from that
//! directory only (no absolute paths, no `..`) to the given URLs, so a job
//! cannot send anything else in the container anywhere. The output is the
//! exit code, the time, the tails of stdout / stderr and the files the run
//! left. `h3 worker --test-input job.json` runs one job without RunPod.

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const TAIL: usize = 16 * 1024;
const VERSION: &str = concat!("h3-worker/", env!("CARGO_PKG_VERSION"));

struct Env {
    get_job: String,
    post_output: String,
    ping: Option<String>,
    key: String,
    ping_ms: u64,
}

impl Env {
    fn from_env() -> Result<Self> {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let worker =
            var("RUNPOD_POD_ID").unwrap_or_else(|| format!("local-{}", std::process::id()));
        let get_job = var("RUNPOD_WEBHOOK_GET_JOB").ok_or_else(|| {
            anyhow!("RUNPOD_WEBHOOK_GET_JOB is not set: this is not a RunPod serverless worker (use --test-input to run a job locally)")
        })?;
        Ok(Env {
            get_job: get_job.replace("$ID", &worker),
            post_output: var("RUNPOD_WEBHOOK_POST_OUTPUT")
                .ok_or_else(|| anyhow!("RUNPOD_WEBHOOK_POST_OUTPUT is not set"))?
                .replace("$RUNPOD_POD_ID", &worker),
            ping: var("RUNPOD_WEBHOOK_PING").map(|p| p.replace("$RUNPOD_POD_ID", &worker)),
            key: var("RUNPOD_AI_API_KEY").unwrap_or_default(),
            ping_ms: var("RUNPOD_PING_INTERVAL")
                .and_then(|v| v.parse().ok())
                .unwrap_or(10_000),
        })
    }
}

fn join_query(url: &str, q: &str) -> String {
    format!("{url}{}{q}", if url.contains('?') { '&' } else { '?' })
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(300))
        .build()
}

/// Run jobs until killed (RunPod scales the worker down).
pub fn run(test_input: Option<PathBuf>, work_root: PathBuf) -> Result<()> {
    if let Some(p) = test_input {
        let input: Value = serde_json::from_slice(&std::fs::read(&p)?)
            .with_context(|| format!("{} is not JSON", p.display()))?;
        let input = input.get("input").cloned().unwrap_or(input);
        let r = run_job("local", &input, &work_root);
        println!("{}", serde_json::to_string_pretty(&r)?);
        if r.get("error").is_some() {
            bail!("the job failed");
        }
        return Ok(());
    }
    let env = Env::from_env()?;
    let http = agent();
    let current: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    if let Some(ping) = env.ping.clone() {
        let (http, key, current, every) =
            (http.clone(), env.key.clone(), current.clone(), env.ping_ms);
        std::thread::spawn(move || {
            loop {
                let mut url = join_query(&ping, &format!("runpod_version={VERSION}"));
                if let Some(id) = current.lock().unwrap().clone() {
                    url = join_query(&url, &format!("job_id={id}"));
                }
                if let Err(e) = http
                    .get(&url)
                    .set("Authorization", &key)
                    .timeout(Duration::from_millis(every * 2))
                    .call()
                {
                    eprintln!("worker: ping failed: {e}");
                }
                std::thread::sleep(Duration::from_millis(every));
            }
        });
    }
    eprintln!("worker: {VERSION} waiting for jobs");
    let mut backoff = 1u64;
    loop {
        let url = join_query(&env.get_job, "job_in_progress=0");
        let jobs = match http.get(&url).set("Authorization", &env.key).call() {
            Ok(r) if r.status() == 204 => Vec::new(),
            Ok(r) => {
                let mut body = String::new();
                r.into_reader().take(64 << 20).read_to_string(&mut body)?;
                match serde_json::from_str::<Value>(&body) {
                    Ok(Value::Array(v)) => v,
                    Ok(v @ Value::Object(_)) => vec![v],
                    _ => Vec::new(),
                }
            }
            Err(ureq::Error::Status(400, _)) => Vec::new(),
            Err(ureq::Error::Status(429, _)) => {
                backoff = (backoff * 2).min(30);
                Vec::new()
            }
            Err(e) => {
                eprintln!("worker: job-take failed: {e}");
                backoff = (backoff * 2).min(30);
                Vec::new()
            }
        };
        if jobs.is_empty() {
            std::thread::sleep(Duration::from_secs(backoff));
            continue;
        }
        backoff = 1;
        for job in jobs {
            let Some(id) = job["id"].as_str().map(str::to_string) else {
                eprintln!("worker: a job without an id: skipped");
                continue;
            };
            *current.lock().unwrap() = Some(id.clone());
            eprintln!("worker: job {id} started");
            let result = run_job(&id, &job["input"], &work_root);
            let url = join_query(&env.post_output.replace("$ID", &id), "isStream=false");
            let body = serde_json::to_string(&result)?;
            let mut sent = false;
            for attempt in 0..3 {
                match http
                    .post(&url)
                    .set("Authorization", &env.key)
                    .set("Content-Type", "application/x-www-form-urlencoded")
                    .send_string(&body)
                {
                    Ok(_) => {
                        sent = true;
                        break;
                    }
                    Err(e) => {
                        eprintln!("worker: posting job {id}'s result failed: {e}");
                        std::thread::sleep(Duration::from_secs(1 << attempt));
                    }
                }
            }
            eprintln!(
                "worker: job {id} {} ({})",
                if result.get("error").is_some() {
                    "failed"
                } else {
                    "done"
                },
                if sent {
                    "result sent"
                } else {
                    "result NOT sent"
                }
            );
            *current.lock().unwrap() = None;
        }
    }
}

/// A file name inside the job directory: relative, no `..`, no root.
fn job_path(dir: &Path, name: &str) -> Result<PathBuf> {
    let p = Path::new(name);
    if name.is_empty() || !p.components().all(|c| matches!(c, Component::Normal(_))) {
        bail!("`{name}`: uploads name files inside the job directory (relative, no `..`)");
    }
    Ok(dir.join(p))
}

/// Run one job; never panics, errors land in the result's `error`.
pub fn run_job(id: &str, input: &Value, work_root: &Path) -> Value {
    let t0 = Instant::now();
    let safe: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let dir = work_root.join(format!("job-{safe}"));
    let res = (|| -> Result<Value> {
        let args: Vec<String> = input["args"]
            .as_array()
            .ok_or_else(|| {
                anyhow!(
                    "input.args: a list of h3 arguments, e.g. [\"check\", \"--backend\", \"cuda\"]"
                )
            })?
            .iter()
            .map(|a| {
                a.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("input.args holds only strings"))
            })
            .collect::<Result<_>>()?;
        match args.first().map(String::as_str) {
            None => bail!("input.args is empty"),
            Some("worker" | "serve") => bail!("`{}` does not return: not a job", args[0]),
            _ => {}
        }
        let uploads: Vec<(String, String)> = match &input["uploads"] {
            Value::Null => Vec::new(),
            Value::Object(m) => m
                .iter()
                .map(|(k, v)| {
                    Ok((
                        k.clone(),
                        v.as_str()
                            .ok_or_else(|| anyhow!("uploads.{k}: a URL"))?
                            .to_string(),
                    ))
                })
                .collect::<Result<_>>()?,
            _ => bail!("input.uploads: an object of file name → PUT URL"),
        };
        for (name, _) in &uploads {
            job_path(&dir, name)?;
        }
        let timeout = input["timeout_s"].as_u64().map(Duration::from_secs);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;

        let exe = std::env::current_exe()?;
        let mut child = Command::new(exe)
            .args(&args)
            .current_dir(&dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("start h3")?;
        let tee = |r: Box<dyn Read + Send>, tag: &'static str| {
            std::thread::spawn(move || {
                let mut tail = String::new();
                for line in BufReader::new(r).lines().map_while(|l| l.ok()) {
                    eprintln!("[{tag}] {line}");
                    tail.push_str(&line);
                    tail.push('\n');
                    if tail.len() > 2 * TAIL {
                        let cut = tail.len() - TAIL;
                        let cut = (cut..tail.len())
                            .find(|&i| tail.is_char_boundary(i))
                            .unwrap_or(cut);
                        tail.drain(..cut);
                    }
                }
                if tail.len() > TAIL {
                    let cut = tail.len() - TAIL;
                    let cut = (cut..tail.len())
                        .find(|&i| tail.is_char_boundary(i))
                        .unwrap_or(cut);
                    tail.drain(..cut);
                }
                tail
            })
        };
        let out = tee(Box::new(child.stdout.take().unwrap()), "out");
        let err = tee(Box::new(child.stderr.take().unwrap()), "err");
        let mut timed_out = false;
        let status = loop {
            if let Some(s) = child.try_wait()? {
                break s;
            }
            if timeout.is_some_and(|t| t0.elapsed() > t) {
                let _ = child.kill();
                timed_out = true;
                break child.wait()?;
            }
            std::thread::sleep(Duration::from_millis(200));
        };
        let (stdout, stderr) = (
            out.join().unwrap_or_default(),
            err.join().unwrap_or_default(),
        );
        let run_s = t0.elapsed().as_secs_f64();

        let mut files = serde_json::Map::new();
        for e in std::fs::read_dir(&dir)?.flatten() {
            if let Ok(m) = e.metadata()
                && m.is_file()
            {
                files.insert(e.file_name().to_string_lossy().into_owned(), json!(m.len()));
            }
        }
        let http = agent();
        let mut uploaded = serde_json::Map::new();
        for (name, url) in &uploads {
            let p = job_path(&dir, name)?;
            let r = (|| -> Result<u64> {
                let f =
                    std::fs::File::open(&p).with_context(|| format!("{name} was not written"))?;
                let n = f.metadata()?.len();
                http.put(url)
                    .set("Content-Length", &n.to_string())
                    .timeout(Duration::from_secs(1800))
                    .send(f)
                    .map_err(|e| anyhow!("PUT {name}: {e}"))?;
                Ok(n)
            })();
            uploaded.insert(
                name.clone(),
                match r {
                    Ok(n) => json!({"bytes": n}),
                    Err(e) => json!({"error": format!("{e:#}")}),
                },
            );
        }
        let code = status.code();
        let mut output = json!({
            "exit_code": code, "seconds": run_s, "timed_out": timed_out,
            "stdout_tail": stdout, "stderr_tail": stderr,
            "files": files, "uploads": uploaded,
        });
        let upload_failed = uploaded.values().any(|v| v.get("error").is_some());
        if code != Some(0) || timed_out || upload_failed {
            let why = if timed_out {
                "h3 timed out".to_string()
            } else if code != Some(0) {
                format!("h3 exited with {code:?}")
            } else {
                "an upload failed".to_string()
            };
            output["error_reason"] = json!(why);
            return Ok(json!({"output": output, "error": why}));
        }
        Ok(json!({"output": output}))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    res.unwrap_or_else(|e| json!({"error": format!("{e:#}")}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uploads_stay_in_the_job_directory() {
        let d = Path::new("/tmp/job");
        assert!(job_path(d, "clip.y4m").is_ok());
        assert!(job_path(d, "out/clip.y4m").is_ok());
        for bad in ["", "/etc/passwd", "../x", "a/../../x", "./x"] {
            assert!(job_path(d, bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn bad_inputs_are_job_errors() {
        let root = std::env::temp_dir().join("h3-worker-test");
        for input in [
            json!({}),
            json!({"args": []}),
            json!({"args": [1]}),
            json!({"args": ["serve"]}),
            json!({"args": ["check"], "uploads": {"../x": "http://x"}}),
        ] {
            let r = run_job("t", &input, &root);
            assert!(r["error"].is_string(), "{input}: {r}");
        }
        assert_eq!(join_query("http://a/b?x=1", "y=2"), "http://a/b?x=1&y=2");
        assert_eq!(join_query("http://a/b", "y=2"), "http://a/b?y=2");
    }
}
