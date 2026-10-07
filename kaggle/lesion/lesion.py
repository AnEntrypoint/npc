#!/usr/bin/env python3
"""Lesion sweep: zero one observation input at a time and measure how much the policy loses.

This is the cheap way to ask whether v5 content is load-bearing. A lesion of
input X costs the genome only the information X carried, so if zeroing
hostileType or bossPhase or tradeNeed changes nothing, the learner never used
that content -- which is the same question the r68/r69 monoculture work kept
hitting, and it needs a full eval batch per lesion, i.e. CPU that stalls a live
GPU training job locally.

Reads payload/lesion.json:

  {"seeds": "7,8,9,10", "worlds": "3", "max": 1,
   "lesions": "hostileType,coverNear,bossPhase,perkTier,tradeNeed"}

Writes out/lesion.json, one row per (candidate, lesion) including the unlesioned
control, and prints a delta-against-control table.
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

DEFAULT_CFG = {
    "suites": "bots", "seeds": "7,8,9,10", "worlds": "3", "max": 1,
    "lesions": "hostileType,coverNear,bossPhase,perkTier,tradeNeed,"
               "bossResistsMelee,bossResistsMage,lvlMelee,lvlMage,"
               "woodPrice,greatProgress,weakerPlayerN",
}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def log(*a):
    print(*a, flush=True)


def find_dataset():
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
    return [p for _m, p in pool[: int(cfg.get("max", 1))]]


def run(genome, cfg, lesion, tag):
    name = "%s.%s" % (tag, lesion or "control")
    cmd = ("cd %s && node tools/evalsuite.mjs --game=%s --suites=%s --seeds=%s --worlds=%s"
           " --periods=1 --jobs=1 --out=%s %s"
           % (ROOT, cfg.get("game", "realm"), cfg["suites"], cfg["seeds"], cfg["worlds"],
              os.path.join(OUT, name + ".json"), genome))
    if lesion:
        cmd += " --lesion=" + lesion
    t0 = time.time()
    r = sh(cmd)
    return {"tag": tag, "lesion": lesion or "control", "genome": genome, "exit": r.returncode,
            "seconds": round(time.time() - t0, 1), "stdout": r.stdout, "stderr": r.stderr[-2000:]}


def score(res, cfg):
    path = os.path.join(OUT, "%s.%s.json" % (res["tag"], res["lesion"]))
    if res["exit"] != 0 or not os.path.isfile(path):
        return None
    with open(path) as f:
        j = json.load(f)
    suite = (j.get("suites") or {}).get(cfg["suites"].split(",")[0]) or {}
    m = suite.get("summary") or {}
    val = lambda k: (m.get(k) or {}).get("mean")
    sd = lambda k: (m.get(k) or {}).get("sd")
    return {"name": res["tag"], "lesion": res["lesion"], "seconds": res["seconds"],
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
    for p in (os.path.join(ROOT, "lesion.json"), os.path.join(HERE, "lesion.json")):
        if os.path.isfile(p):
            with open(p) as f:
                cfg.update(json.load(f))
            log("config from " + p)
            break

    pool = candidates(ROOT, cfg)
    if not pool:
        log("FATAL: no runs/*-base in the payload")
        sys.exit(1)
    lesions = [""] + [s.strip() for s in str(cfg["lesions"]).split(",") if s.strip()]
    log("cpu_count=%s | candidates=%d | lesions=%d | cfg=%s"
        % (os.cpu_count(), len(pool), len(lesions), json.dumps(cfg)))

    smoke = run(pool[0], cfg, "", "smoke")
    log("## smoke exit=%s %ss\n%s" % (smoke["exit"], smoke["seconds"], smoke["stdout"].strip()))
    if smoke["exit"] != 0:
        log("SMOKE FAILED, aborting\nSTDERR:\n" + smoke["stderr"])
        sys.exit(1)

    jobs = [(g, os.path.basename(g), l) for g in pool for l in lesions]
    workers = max(1, min(len(jobs), os.cpu_count() or 4))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda j: run(j[0], cfg, j[2], j[1]), jobs))
    log("\nran %d evals in %.1fs" % (len(results), time.time() - t0))

    rows = []
    for res in results:
        s = score(res, cfg)
        if s is None:
            log("## %s %s FAILED exit=%s\n%s" % (res["tag"], res["lesion"], res["exit"], res["stderr"]))
        else:
            rows.append(s)
    if not rows:
        log("FATAL: every lesion run failed")
        sys.exit(1)

    byName = {}
    for s in rows:
        byName.setdefault(s["name"], {})[s["lesion"]] = s
    log("\n%-24s %10s %10s %8s" % ("candidate/lesion", "rate", "vs control", "life"))
    for name, per in byName.items():
        ctl = per.get("control")
        base = ctl["rate"] if ctl and ctl["rate"] is not None else None
        for s in sorted(per.values(), key=lambda r: r["lesion"]):
            d = ""
            if base is not None and s["rate"] is not None and s["lesion"] != "control":
                d = "%+.1f%%" % (100 * (s["rate"] - base) / base) if base else "n/a"
            log("%-24s %10s %10s %8s" % (
                s["lesion"],
                "n/a" if s["rate"] is None else "%.5f" % s["rate"],
                d,
                "n/a" if s["life"] is None else "%.0f" % s["life"]))

    with open(os.path.join(OUT, "lesion.json"), "w") as f:
        json.dump({"cfg": cfg, "rows": rows}, f, indent=1)
    log("\nDONE")


if __name__ == "__main__":
    main()
