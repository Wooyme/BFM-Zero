"""Run a command under a memory watchdog: kill its whole process group when system memory use passes a limit.

    uv run python scripts/mem_guard.py --limit 90 --log mem.log -- python -m experiments.stable_wrist ...

Every interval it logs the system memory use, the total RSS of the command's process tree and its largest processes;
above limit percent it kills the process group (the command, its workers and its Isaac subprocesses) and exits 137.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import psutil


def tree(pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(pid)
        return [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def rss(p: psutil.Process) -> int:
    try:
        return p.memory_info().rss
    except psutil.Error:
        return 0


def describe(p: psutil.Process) -> str:
    try:
        cmd = " ".join(p.cmdline())[-60:]
    except psutil.Error:
        cmd = "?"
    return f"{p.pid}:{rss(p) / 2**30:.1f}G[{cmd}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=float, default=90.0, help="percent of system memory in use")
    parser.add_argument("--interval", type=float, default=2.0, help="s between log lines (memory is checked every 0.2 s)")
    parser.add_argument("--log", default=None, help="file of the memory log (default: stderr)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    log = open(args.log, "a", buffering=1) if args.log else sys.stderr
    proc = subprocess.Popen(command, start_new_session=True)  # its own process group, killed as a whole
    start, last, peak = time.time(), 0.0, 0.0
    while proc.poll() is None:
        used = psutil.virtual_memory().percent
        peak = max(peak, used)
        now = time.time()
        if used > args.limit or now - last >= args.interval:
            procs = tree(proc.pid)
            top = sorted(procs, key=rss, reverse=True)[:4]
            log.write(f"[mem {now - start:7.0f}s] system {used:.1f}% (peak {peak:.1f}%), tree "
                      f"{sum(map(rss, procs)) / 2**30:.1f}G in {len(procs)} procs: {' '.join(map(describe, top))}\n")
            last = now
        if used > args.limit:
            log.write(f"[mem] system memory {used:.1f}% > {args.limit}%: killing process group {proc.pid}\n")
            for p in tree(proc.pid):  # descendants in their own session too
                try:
                    p.kill()
                except psutil.Error:
                    pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            sys.exit(137)
        time.sleep(0.2)
    log.write(f"[mem] command exited {proc.returncode}, peak system memory {peak:.1f}%\n")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
