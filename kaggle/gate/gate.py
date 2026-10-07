#!/usr/bin/env python3
"""Score the top candidates once per forced attack style, so style balance costs no local CPU.

The open question on realm v5 is monoculture: trained learners end up almost
purely ranged, so melee and mage content never fires. tools/evalsuite.mjs takes
--gate=N, which forces execStyle = (chosen + N) % 3 without touching movement,
so running the same genome at gate 0/1/2 separates "the policy cannot melee"
from "melee is underpowered". That is 3x the eval work per candidate, which is
exactly the kind of CPU batch that stalls a live GPU training job locally.

Reads payload/gate.json:

  {"suites": "bots", "seeds": "7,8,9,10", "worlds": "3", "gates": "0,1,2", "max": 4}

and writes out/gate.json -- one row per (candidate, gate) -- plus the same table
printed to the log, so `kaggle kernels output` and the kernel log both carry it.
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

DEFAULT_CFG = {"suites": "bots", "seeds": "7,8,9,10", "worlds": "3", "gates": "0,1,2", "max": 4}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


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
    limit = int(cfg.get("max", 4))
    return [p for _m, p in pool[:limit]]


def run(genome, cfg, gate, tag):
    name = "%s.g%s" % (tag, gate)
    cmd = ("cd %s && node tools/evalsuite.mjs --game=%s --suites=%s --seeds=%s --worlds=%s"
           " --periods=1 --jobs=1 --gate=%s --out=%s %s"
           % (ROOT, cfg.get("game", "realm"), cfg["suites"], cfg["seeds"], cfg["worlds"],
              gate, os.path.join(OUT, name + ".json"), genome))
    t0 = time.time()
    r = sh(cmd)
    return {"tag": tag, "gate": gate, "genome": genome, "exit": r.returncode,
            "seconds": round(time.time() - t0, 1), "stdout": r.stdout, "stderr": r.stderr[-2000:]}


def score(res, cfg):
    path = os.path.join(OUT, "%s.g%s.json" % (res["tag"], res["gate"]))
    if res["exit"] != 0 or not os.path.isfile(path):
        return None
    with open(path) as f:
        j = json.load(f)
    suite = (j.get("suites") or {}).get(cfg["suites"].split(",")[0]) or {}
    m = suite.get("summary") or {}
    val = lambda k: (m.get(k) or {}).get("mean")
    sd = lambda k: (m.get(k) or {}).get("sd")
    return {"name": res["tag"], "gate": res["gate"], "seconds": res["seconds"],
            "rate": val("rate"), "rateSd": sd("rate"), "ratio": val("ratio"),
            "life": val("life"), "toolTier": val("toolTier")}


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
    for p in (os.path.join(ROOT, "gate.json"), os.path.join(HERE, "gate.json")):
        if os.path.isfile(p):
            with open(p) as f:
                cfg.update(json.load(f))
            log("config from " + p)
            break

    pool = candidates(ROOT, cfg)
    if not pool:
        log("FATAL: no runs/*-base in the payload")
        sys.exit(1)
    gates = [g.strip() for g in str(cfg["gates"]).split(",") if g.strip() != ""]
    log("cpu_count=%s | candidates=%d | gates=%s | cfg=%s"
        % (os.cpu_count(), len(pool), gates, json.dumps(cfg)))

    smoke = run(pool[0], cfg, gates[0], "smoke")
    log("## smoke exit=%s %ss\n%s" % (smoke["exit"], smoke["seconds"], smoke["stdout"].strip()))
    if smoke["exit"] != 0:
        log("SMOKE FAILED, aborting\nSTDERR:\n" + smoke["stderr"])
        sys.exit(1)

    jobs = [(g, os.path.basename(g), gate) for g in pool for gate in gates]
    workers = max(1, min(len(jobs), os.cpu_count() or 4))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda j: run(j[0], cfg, j[2], j[1]), jobs))
    log("\nran %d evals in %.1fs" % (len(results), time.time() - t0))

    rows = []
    for res in results:
        s = score(res, cfg)
        if s is None:
            log("## %s g%s FAILED exit=%s\n%s" % (res["tag"], res["gate"], res["exit"], res["stderr"]))
        else:
            rows.append(s)
    if not rows:
        log("FATAL: every gate run failed")
        sys.exit(1)

    log("\n%-24s %4s %10s %8s %8s %8s" % ("candidate", "gate", "rate", "ratio", "life", "tool"))
    for s in sorted(rows, key=lambda r: (r["name"], r["gate"])):
        log("%-24s %4s %10s %8s %8s %8s" % (
            s["name"], s["gate"],
            "n/a" if s["rate"] is None else "%.5f" % s["rate"],
            "n/a" if s["ratio"] is None else "%.2f" % s["ratio"],
            "n/a" if s["life"] is None else "%.0f" % s["life"],
            "n/a" if s["toolTier"] is None else "%.2f" % s["toolTier"]))

    log("\n%-24s %10s %10s %10s   melee/range/mage share of the best gate" % ("candidate", "g0", "g1", "g2"))
    byName = {}
    for s in rows:
        byName.setdefault(s["name"], {})[s["gate"]] = s["rate"]
    for name, per in byName.items():
        rs = [per.get(g) for g in gates]
        log("%-24s %10s %10s %10s" % (name,
            *(("n/a" if r is None else "%.5f" % r) for r in rs)))

    with open(os.path.join(OUT, "gate.json"), "w") as f:
        json.dump({"cfg": cfg, "rows": rows}, f, indent=1)
    log("\nDONE")


if __name__ == "__main__":
    main()
