#!/usr/bin/env python3
"""Pick the strongest champion genome and emit it, so the choice costs zero local CPU.

tools/publish.mjs does this locally, but a CPU eval batch and a live GPU training
job stall each other on one machine, so while the GPU trains the publish decision
is simply blocked. This kernel is that decision moved to Kaggle: it scores every
candidate in the payload with tools/evalsuite.mjs, ranks them, and writes

  out/champion.json       the winner, ready to commit
  out/champion.meta.json  its score, its source and the runners-up
  out/ranked.json         every candidate, ranked

so `kaggle kernels output` fetches the answer instead of the local box computing
it. It replaces champion.json only when the winner beats the standing champion by
more than one sd -- same rule as tools/publish.mjs.

Config is payload/publish.json (this kernel reads publish.json, cpu-eval reads
jobs.json, so the two kernels can share one dataset without colliding):

  {"suites": "bots", "seeds": "7,8,9,10,11,12", "worlds": "4", "max": 16,
   "popWeights": "1,1,1,1"}

Missing config means "score every runs/*-base in the payload, newest first".
"""

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

WORK = "/kaggle/working"
INPUT_ROOT = "/kaggle/input"
ROOT = os.path.join(WORK, "train")
OUT = os.path.join(WORK, "out")
HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CFG = {"suites": "bots", "seeds": "7,8,9,10,11,12", "worlds": "4", "max": 16}


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def log(*a):
    print(*a, flush=True)


def find_dataset():
    """Kaggle mounts datasets at /kaggle/input/<slug> on older images and
    /kaggle/input/datasets/<user>/<slug> on current ones, so recognise the
    payload by content rather than by path."""
    if not os.path.isdir(INPUT_ROOT):
        return None
    cands = []
    for dirpath, dirnames, _files in os.walk(INPUT_ROOT):
        if dirpath[len(INPUT_ROOT):].count(os.sep) > 3:
            dirnames[:] = []
            continue
        if os.path.isfile(os.path.join(dirpath, "tools", "evalsuite.mjs")):
            cands.append(dirpath)
    cands.sort(key=len)
    return cands[0] if cands else None


def candidates(root, cfg):
    runs = os.path.join(root, "runs")
    pool = []
    if os.path.isdir(runs):
        for f in os.listdir(runs):
            if not f.endswith("-base") or f.endswith("-best"):
                continue
            p = os.path.join(runs, f)
            if os.path.isfile(p):
                pool.append((os.path.getmtime(p), p))
    pool.sort(key=lambda t: -t[0])
    pool = [p for _m, p in pool[: int(cfg.get("max", 16))]]
    standing = os.path.join(root, "champion.json")
    if os.path.isfile(standing):
        pool.append(standing)
    seen = set()
    return [p for p in pool if not (os.path.basename(p) in seen or seen.add(os.path.basename(p)))]


def cmd_for(genome, cfg, tag, name):
    parts = [f"cd {ROOT} && node tools/evalsuite.mjs",
             f" --game={cfg.get('game', 'realm')}",
             f" --suites={cfg['suites']}",
             f" --seeds={cfg['seeds']}",
             f" --worlds={cfg['worlds']}",
             " --periods=1 --jobs=1"]
    if cfg.get("popWeights"):
        parts.append(' --popWeights="' + cfg["popWeights"] + '"')
    parts.append(f" --out={os.path.join(OUT, name + '.json')}")
    parts.append(" " + genome)
    return "".join(parts)


def run(genome, cfg, tag, name):
    cmd = cmd_for(genome, cfg, tag, name)
    t0 = time.time()
    r = sh(cmd)
    return {"name": name, "genome": genome, "exit": r.returncode,
            "seconds": round(time.time() - t0, 1),
            "stdout": r.stdout, "stderr": r.stderr[-2000:]}


def score(res, cfg):
    path = os.path.join(OUT, res["name"] + ".json")
    if res["exit"] != 0 or not os.path.isfile(path):
        return None
    with open(path) as f:
        j = json.load(f)
    suite = (j.get("suites") or {}).get(cfg["suites"].split(",")[0]) or {}
    m = suite.get("summary") or {}
    mean = lambda k: (m.get(k) or {}).get("mean")
    sd = lambda k: (m.get(k) or {}).get("sd")
    return {"name": res["name"], "source": res["genome"], "seconds": res["seconds"],
            "rate": mean("rate"), "rateSd": sd("rate"), "ratio": mean("ratio"),
            "life": mean("life"), "toolTier": mean("toolTier")}


def main():
    ds = find_dataset()
    if not ds:
        log("FATAL: no dataset with tools/evalsuite.mjs under /kaggle/input")
        sys.exit(1)
    log("dataset: " + ds)
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    shutil.copytree(ds, ROOT)
    os.makedirs(OUT, exist_ok=True)

    cfg = dict(DEFAULT_CFG)
    for p in (os.path.join(ROOT, "publish.json"), os.path.join(HERE, "publish.json")):
        if os.path.isfile(p):
            with open(p) as f:
                cfg.update(json.load(f))
            log(f"config from {p}")
            break

    pool = candidates(ROOT, cfg)
    if not pool:
        log("FATAL: no runs/*-base and no champion.json in the payload")
        sys.exit(1)
    log(f"cpu_count={os.cpu_count()} | candidates={len(pool)} | cfg={json.dumps(cfg)}")

    # One world on one seed first: a broken invocation costs seconds here and the
    # whole sweep otherwise.
    smoke = run(pool[0], cfg, "smoke", "smoke")
    log(f"## smoke exit={smoke['exit']} {smoke['seconds']}s\n{smoke['stdout'].strip()}")
    if smoke["exit"] != 0:
        log("SMOKE FAILED, aborting\nSTDERR:\n" + smoke["stderr"])
        sys.exit(1)

    jobs = [(g, os.path.basename(g)) for g in pool]
    workers = max(1, min(len(jobs), os.cpu_count() or 4))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool_:
        results = list(pool_.map(lambda j: run(j[0], cfg, "", j[1]), jobs))
    log(f"\nscored {len(results)} candidates in {time.time() - t0:.1f}s")

    scored = []
    for res in results:
        s = score(res, cfg)
        if s is None:
            log(f"## {res['name']} FAILED exit={res['exit']}\n{res['stderr']}")
        else:
            scored.append(s)
    scored.sort(key=lambda s: -(s["rate"] if s["rate"] is not None else -1e9))

    log("\n%-28s %10s %8s %8s %8s" % ("candidate", "rate", "ratio", "life", "tool"))
    for s in scored:
        log("%-28s %10s %8s %8s %8s" % (
            s["name"],
            "n/a" if s["rate"] is None else f"{s['rate']:.5f}",
            "n/a" if s["ratio"] is None else f"{s['ratio']:.2f}",
            "n/a" if s["life"] is None else f"{s['life']:.0f}",
            "n/a" if s["toolTier"] is None else f"{s['toolTier']:.2f}"))

    if not scored:
        log("FATAL: every candidate failed")
        sys.exit(1)

    best = scored[0]
    standing = next((s for s in scored if os.path.basename(s["source"]) == "champion.json"), None)
    log(f"\nbest: {best['name']} rate {best['rate']:.5f}")
    if standing is not None:
        beats = best["rate"] > standing["rate"] + (best.get("rateSd") or 0)
        log(f"standing champion: rate {standing['rate']:.5f} -> {'REPLACE' if beats else 'KEEP'}")
    else:
        beats = True
        log("no standing champion.json in the payload -> emitting the best candidate")

    with open(os.path.join(OUT, "ranked.json"), "w") as f:
        json.dump(scored, f, indent=1)
    if beats:
        src = os.path.join(ROOT, best["source"])
        shutil.copyfile(src, os.path.join(OUT, "champion.json"))
        with open(os.path.join(OUT, "champion.meta.json"), "w") as f:
            json.dump({"source": best["source"], "rate": best["rate"], "ratio": best["ratio"],
                       "life": best["life"], "toolTier": best["toolTier"],
                       "seeds": cfg["seeds"], "worlds": cfg["worlds"],
                       "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "runnersUp": [{"source": s["source"], "rate": s["rate"]} for s in scored[1:5]]},
                      f, indent=1)
        log("wrote out/champion.json from " + best["source"])
    log("\nDONE")


if __name__ == "__main__":
    main()
