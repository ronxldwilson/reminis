"""Fine-tune 10 LoRA adapters on Qwen2.5-0.5B-Instruct, one per domain.

Usage:
    uv run experiments/multidomain/finetune_all.py

Resume-safe: re-run anytime — completed domains are skipped,
partial runs resume from the latest checkpoint.
"""

import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASE_MODEL = str(SCRIPT_DIR / "base_model")
DATASETS_DIR = SCRIPT_DIR / "datasets"
ADAPTERS_DIR = SCRIPT_DIR / "adapters"
LOG_DIR = SCRIPT_DIR / "logs"

DOMAINS = ["code", "creative", "finance", "history", "legal",
           "math", "medical", "reasoning", "science", "sql"]

ITERS = 200
BATCH_SIZE = 2
LR = 1e-5
NUM_LAYERS = 8
MAX_SEQ_LEN = 512
SAVE_EVERY = 50

# --- ANSI helpers ---
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
GRAY = "\033[90m"
WHITE = "\033[97m"
CLEAR_SCREEN = "\033[2J\033[H"


def _fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def find_latest_checkpoint(adapter_dir):
    checkpoints = sorted(adapter_dir.glob("adapters-*.safetensors"))
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    step = int(latest.stem.split("-")[-1])
    return str(latest), step


class Display:
    def __init__(self, domains):
        self.domains = domains
        self.status = {d: "pending" for d in domains}
        self.final_loss = {}
        self.final_time = {}
        self.start_time = time.time()
        self.current_domain = None
        self.current_iter = 0
        self.current_train_loss = None
        self.current_val_loss = None
        self.current_it_sec = None
        self.current_tok_sec = None
        self.current_peak_mem = None
        self.current_lr = None
        self.current_tokens = None
        self.current_checkpoint = None
        self.train_log = []  # last N training lines for the active domain

    def set_active(self, domain, status="running"):
        self.current_domain = domain
        self.status[domain] = status
        self.current_iter = 0
        self.current_train_loss = None
        self.current_val_loss = None
        self.current_it_sec = None
        self.current_tok_sec = None
        self.current_peak_mem = None
        self.current_lr = None
        self.current_tokens = None
        self.current_checkpoint = None
        self.train_log = []
        self.render()

    def finish(self, domain, status, loss=None, elapsed=None):
        self.status[domain] = status
        if loss is not None:
            self.final_loss[domain] = loss
        if elapsed is not None:
            self.final_time[domain] = elapsed
        self.current_domain = None
        self.train_log = []
        self.render()

    def parse_line(self, line):
        raw = line.strip()
        if not raw:
            return

        # "Iter 5: Train loss 1.356, Learning Rate 1.000e-05, It/sec 3.242, Tokens/sec 1004.966, Trained Tokens 1630, Peak mem 2.192 GB"
        m = re.match(r"Iter (\d+): Train loss ([\d.]+), Learning Rate ([\d.e+-]+), It/sec ([\d.]+), Tokens/sec ([\d.]+), Trained Tokens (\d+), Peak mem ([\d.]+) GB", raw)
        if m:
            self.current_iter = int(m.group(1))
            self.current_train_loss = float(m.group(2))
            self.current_lr = m.group(3)
            self.current_it_sec = float(m.group(4))
            self.current_tok_sec = float(m.group(5))
            self.current_tokens = int(m.group(6))
            self.current_peak_mem = m.group(7)
            self._log(f"  {WHITE}iter {self.current_iter:>4d}/{ITERS}{RESET}  loss={YELLOW}{self.current_train_loss:.4f}{RESET}  lr={self.current_lr}  {self.current_it_sec:.1f} it/s  {self.current_tok_sec:.0f} tok/s  {self.current_tokens} tokens  mem={self.current_peak_mem}GB")
            return

        # "Iter 5: Val loss 1.347, Val took 3.821s"
        m = re.match(r"Iter (\d+): Val loss ([\d.]+), Val took ([\d.]+)s", raw)
        if m:
            self.current_val_loss = float(m.group(2))
            val_time = m.group(3)
            self._log(f"  {CYAN}──── validation ──── loss={self.current_val_loss:.4f}  took {val_time}s{RESET}")
            return

        # "Saved adapter weights" or checkpoint saves
        if "Saved" in raw:
            self.current_checkpoint = raw
            self._log(f"  {GREEN}💾 {raw}{RESET}")
            return

        # progress bars (Calculating loss) — skip the noisy partial updates
        if "Calculating loss" in raw:
            return

        # everything else: show it
        if "Loading" in raw or "Training" in raw or "Starting" in raw:
            self._log(f"  {BOLD}{raw}{RESET}")
        elif "Trainable" in raw:
            self._log(f"  {CYAN}{raw}{RESET}")
        elif "WARNING" in raw:
            self._log(f"  {YELLOW}⚠ {raw}{RESET}")
        elif raw:
            self._log(f"  {DIM}{raw}{RESET}")

    def _log(self, formatted_line):
        self.train_log.append(formatted_line)
        if len(self.train_log) > 20:
            self.train_log = self.train_log[-20:]
        self.render()

    def render(self):
        icons = {"pending": "○", "running": "▶", "done": "✓", "failed": "✗", "skipped": "–"}
        colors = {"pending": GRAY, "running": YELLOW, "done": GREEN, "failed": RED, "skipped": GRAY}

        elapsed_total = time.time() - self.start_time
        done_count = sum(1 for s in self.status.values() if s in ("done", "skipped"))
        running = sum(1 for s in self.status.values() if s == "running")
        total = len(self.domains)

        lines = []
        lines.append(f"{BOLD}{'═' * 62}{RESET}")
        lines.append(f"{BOLD}  MULTI-DOMAIN LoRA FINE-TUNING  │  {_fmt_time(elapsed_total)}{RESET}")
        lines.append(f"{BOLD}  Qwen2.5-0.5B-Instruct  │  {ITERS} iters/domain  │  LoRA r=8{RESET}")
        lines.append(f"{BOLD}{'═' * 62}{RESET}")

        # progress bar
        bar_w = 40
        filled = int(bar_w * done_count / total)
        bar = "█" * filled + "░" * (bar_w - filled)
        lines.append(f"  [{bar}] {done_count}/{total}")
        lines.append("")

        # domain list
        for d in self.domains:
            s = self.status[d]
            icon = icons[s]
            c = colors[s]
            parts = f"  {c}{icon}{RESET} {c}{d:12s}{RESET}"

            if s == "done":
                t = f" {DIM}{_fmt_time(self.final_time.get(d, 0))}{RESET}" if d in self.final_time else ""
                l = f"  loss={GREEN}{self.final_loss[d]:.4f}{RESET}" if d in self.final_loss else ""
                parts += t + l
            elif s == "running" and self.current_iter > 0:
                pct = self.current_iter / ITERS * 100
                mini_bar_w = 12
                mini_filled = int(mini_bar_w * self.current_iter / ITERS)
                mini_bar = "━" * mini_filled + "╌" * (mini_bar_w - mini_filled)
                parts += f" {YELLOW}{mini_bar}{RESET} {pct:.0f}%"
                if self.current_train_loss is not None:
                    parts += f"  loss={YELLOW}{self.current_train_loss:.4f}{RESET}"

            lines.append(parts)

        # active training detail
        if self.train_log:
            lines.append("")
            lines.append(f"  {BOLD}{'─' * 56}{RESET}")
            lines.append(f"  {BOLD}Training: {self.current_domain}{RESET}")
            lines.append("")
            for tl in self.train_log:
                lines.append(tl)

        # ETA
        completed_real = sum(1 for s in self.status.values() if s == "done")
        if completed_real > 0 and completed_real < total:
            avg = elapsed_total / completed_real
            pending = total - done_count - running
            remaining = avg * pending
            if running and self.current_iter > 0:
                remaining += avg * (1 - self.current_iter / ITERS)
            lines.append("")
            lines.append(f"  {DIM}ETA: ~{_fmt_time(remaining)} remaining{RESET}")

        lines.append("")

        sys.stdout.write(CLEAR_SCREEN)
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()


def train_domain(domain, display):
    data_dir = DATASETS_DIR / domain
    adapter_dir = ADAPTERS_DIR / domain

    if not (data_dir / "train.jsonl").exists():
        display.finish(domain, "skipped")
        return "skipped"

    if (adapter_dir / "adapters.safetensors").exists():
        display.finish(domain, "done")
        return "done"

    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{domain}.log"

    display.set_active(domain)

    resume_flag = []
    checkpoint = find_latest_checkpoint(adapter_dir)
    if checkpoint:
        resume_flag = ["--resume-adapter-file", checkpoint[0]]

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", BASE_MODEL,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--iters", str(ITERS),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LR),
        "--num-layers", str(NUM_LAYERS),
        "--max-seq-length", str(MAX_SEQ_LEN),
        "--steps-per-report", "10",
        "--steps-per-eval", "50",
        "--save-every", str(SAVE_EVERY),
    ] + resume_flag

    start = time.time()

    with open(log_file, "a") as fh:
        fh.write(f"\n{'=' * 60}\n")
        fh.write(f"[{datetime.now().isoformat()}] Domain: {domain}\n")
        fh.write(f"Command: {' '.join(cmd)}\n")
        fh.write(f"{'=' * 60}\n\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    last_loss = None
    buf = b""
    with open(log_file, "ab") as fh:
        while True:
            chunk = proc.stdout.read(1)
            if not chunk:
                if buf:
                    line = buf.decode("utf-8", errors="replace").strip()
                    fh.write(buf + b"\n")
                    fh.flush()
                    if line:
                        display.parse_line(line)
                        m = re.search(r"Train loss ([\d.]+)", line)
                        if m:
                            last_loss = float(m.group(1))
                break

            if chunk in (b"\n", b"\r"):
                line = buf.decode("utf-8", errors="replace").strip()
                fh.write(buf + b"\n")
                fh.flush()
                if line:
                    display.parse_line(line)
                    m = re.search(r"Train loss ([\d.]+)", line)
                    if m:
                        last_loss = float(m.group(1))
                buf = b""
            else:
                buf += chunk

    proc.wait()
    elapsed = time.time() - start

    if proc.returncode == 0:
        display.finish(domain, "done", loss=last_loss, elapsed=elapsed)
        return "done"
    else:
        display.finish(domain, "failed", elapsed=elapsed)
        return "failed"


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    display = Display(DOMAINS)
    display.render()

    results = {"done": 0, "skipped": 0, "failed": 0}

    for domain in DOMAINS:
        result = train_domain(domain, display)
        results[result] = results.get(result, 0) + 1

    # final screen
    elapsed = time.time() - display.start_time
    display.render()
    print(f"\n{BOLD}  All done in {_fmt_time(elapsed)}{RESET}")
    print(f"  Completed: {GREEN}{results['done']}{RESET}  Skipped: {GRAY}{results['skipped']}{RESET}  Failed: {RED}{results['failed']}{RESET}")
    print(f"\n  Adapters: {ADAPTERS_DIR}")
    print(f"  Logs:     {LOG_DIR}")

    summary = {
        "elapsed_seconds": elapsed,
        "results": results,
        "domains": {d: display.status[d] for d in DOMAINS},
        "final_losses": {d: display.final_loss.get(d) for d in DOMAINS},
        "times_seconds": {d: display.final_time.get(d) for d in DOMAINS},
    }
    with open(SCRIPT_DIR / "finetune_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
