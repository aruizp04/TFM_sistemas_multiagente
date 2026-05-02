#!/usr/bin/env python3
"""
Monitor simple de rendimiento para simulaciones ROS 2 / Open-RMF.

Flujo:
  1. Lanza una tarea RMF con dispatch_loop.
  2. Extrae el task_id devuelto por RMF.
  3. Mide CPU/RAM/GPU del sistema y CPU/RAM de procesos relevantes.
  4. Consulta /fleet_states hasta detectar que el robot ha pasado de tener ese task_id
     a tener task_id vacio.
  5. Guarda CSV, resumen JSON y graficas.

Uso:
  python3 monitor_navigation_performance_simplified.py --label nav2_loop_01
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import select
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError as exc:
    raise SystemExit("Falta psutil. Instala con: python3 -m pip install psutil") from exc


DEFAULT_TASK_CMD = (
    "ros2 run rmf_demos_tasks dispatch_loop "
    "-s turtlebot1_charger -f hall_4 -n 1 --use_sim_time"
)

DEFAULT_PROCESS_KEYWORDS = [
    "ros2",
    "rmf",
    "gz",
    "gazebo",
    "nav2",
    "easynav",
    "rviz",
    "robot_state_publisher",
    "controller_manager",
]


# -----------------------------------------------------------------------------
# Utilidades generales
# -----------------------------------------------------------------------------

def mib(value_bytes: int | float) -> float:
    return round(float(value_bytes) / (1024.0 * 1024.0), 3)


def run_bash(command: str, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )


def write_csv_row(path: Path, row: dict[str, Any]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def safe_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


# -----------------------------------------------------------------------------
# Lectura de rendimiento
# -----------------------------------------------------------------------------

def read_gpu() -> dict[str, float | None]:
    """Lee GPU NVIDIA si existe. Si no existe, devuelve campos None."""
    empty = {
        "gpu_utilization_percent": None,
        "gpu_memory_used_mib": None,
        "gpu_memory_total_mib": None,
        "gpu_temperature_c": None,
        "gpu_power_draw_w": None,
    }

    if run_bash("command -v nvidia-smi >/dev/null 2>&1").returncode != 0:
        return empty

    cmd = (
        "nvidia-smi "
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw "
        "--format=csv,noheader,nounits"
    )

    try:
        result = run_bash(cmd, timeout=2.0)
    except Exception:
        return empty

    if result.returncode != 0 or not result.stdout.strip():
        return empty

    rows: list[list[float]] = []
    for line in result.stdout.splitlines():
        values = [safe_float(v) for v in line.split(",")]
        if len(values) == 5 and all(v is not None for v in values):
            rows.append([float(v) for v in values])

    if not rows:
        return empty

    return {
        "gpu_utilization_percent": sum(r[0] for r in rows) / len(rows),
        "gpu_memory_used_mib": sum(r[1] for r in rows),
        "gpu_memory_total_mib": sum(r[2] for r in rows),
        "gpu_temperature_c": sum(r[3] for r in rows) / len(rows),
        "gpu_power_draw_w": sum(r[4] for r in rows),
    }


def read_cpu_temp() -> float | None:
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        return None

    values = [entry.current for group in temps.values() for entry in group if entry.current]
    return max(values) if values else None


def read_system_sample(start_time: float) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_io_counters()
    net = psutil.net_io_counters()
    freq = psutil.cpu_freq()

    row = {
        "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round(time.monotonic() - start_time, 3),
        "cpu_total_percent": psutil.cpu_percent(interval=None),
        "cpu_per_core_percent": json.dumps(psutil.cpu_percent(interval=None, percpu=True)),
        "ram_used_mib": mib(vm.used),
        "ram_total_mib": mib(vm.total),
        "ram_percent": vm.percent,
        "swap_used_mib": mib(swap.used),
        "swap_total_mib": mib(swap.total),
        "swap_percent": swap.percent,
        "disk_read_mib": mib(disk.read_bytes) if disk else 0.0,
        "disk_write_mib": mib(disk.write_bytes) if disk else 0.0,
        "net_sent_mib": mib(net.bytes_sent) if net else 0.0,
        "net_recv_mib": mib(net.bytes_recv) if net else 0.0,
        "cpu_freq_current_mhz": freq.current if freq else None,
        "cpu_temp_c": read_cpu_temp(),
    }

    row.update(read_gpu())
    return row


def process_matches(proc: psutil.Process, keyword: str) -> bool:
    try:
        text = f"{proc.name()} {' '.join(proc.cmdline())}".lower()
    except Exception:
        return False
    return keyword.lower() in text


def prime_process_cpu() -> None:
    for proc in psutil.process_iter():
        try:
            proc.cpu_percent(interval=None)
        except Exception:
            pass


def read_process_samples(start_time: float, keywords: list[str]) -> list[dict[str, Any]]:
    processes = list(psutil.process_iter())
    rows: list[dict[str, Any]] = []

    for keyword in keywords:
        cpu_sum = 0.0
        rss_sum = 0.0
        vms_sum = 0.0
        pids: list[str] = []
        names: set[str] = set()

        for proc in processes:
            if not process_matches(proc, keyword):
                continue
            try:
                mem = proc.memory_info()
                cpu_sum += proc.cpu_percent(interval=None)
                rss_sum += mib(mem.rss)
                vms_sum += mib(mem.vms)
                pids.append(str(proc.pid))
                names.add(proc.name())
            except Exception:
                continue

        rows.append(
            {
                "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
                "elapsed_s": round(time.monotonic() - start_time, 3),
                "keyword": keyword,
                "process_count": len(pids),
                "cpu_percent_sum": round(cpu_sum, 3),
                "ram_rss_mib_sum": round(rss_sum, 3),
                "ram_vms_mib_sum": round(vms_sum, 3),
                "pids": ";".join(pids),
                "process_names": ";".join(sorted(names)),
            }
        )

    return rows


# -----------------------------------------------------------------------------
# ROS / RMF
# -----------------------------------------------------------------------------

def launch_task(task_cmd: str, ros_setup: str | None) -> subprocess.Popen[str]:
    command = f"source {ros_setup} && {task_cmd}" if ros_setup else task_cmd
    return subprocess.Popen(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )


def drain_output(proc: subprocess.Popen[str], log_path: Path) -> list[str]:
    if proc.stdout is None:
        return []

    lines: list[str] = []
    while True:
        ready, _, _ = select.select([proc.stdout], [], [], 0)
        if not ready:
            break

        line = proc.stdout.readline()
        if not line:
            break

        lines.append(line)
        print(f"[task] {line}", end="")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    return lines


def extract_task_id(lines: list[str]) -> str | None:
    text = "\n".join(lines)
    patterns = [
        r"assigned task_id:\s*\[([^\]]+)\]",
        r"task_id=['\"]([^'\"]+)['\"]",
        r"task_id:\s*([^\s,\]]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            proc.wait(timeout=5)
            return
        except Exception:
            continue


def read_topic_once(topic: str, ros_setup: str | None, timeout_s: float) -> str:
    setup = f"source {ros_setup} && " if ros_setup else ""
    cmd = f"{setup}timeout {timeout_s:.1f}s ros2 topic echo {topic} --once --full-length"
    try:
        return run_bash(cmd, timeout=timeout_s + 2.0).stdout or ""
    except Exception:
        return ""


def robot_block(raw_fleet_states: str, robot_name: str) -> str:
    if robot_name not in raw_fleet_states:
        return ""

    lines = raw_fleet_states.splitlines()
    index = next((i for i, line in enumerate(lines) if robot_name in line), None)
    if index is None:
        return ""

    return "\n".join(lines[max(0, index - 10): min(len(lines), index + 80)])


def parse_robot_status(raw_fleet_states: str, robot_name: str) -> dict[str, Any]:
    block = robot_block(raw_fleet_states, robot_name)
    if not block:
        return {"found": False, "mode": None, "task_id": "", "raw_block": ""}

    mode_match = re.search(r"mode:\s*\n\s*mode:\s*(-?\d+)", block) or re.search(r"\bmode:\s*(-?\d+)", block)
    task_match = re.search(r"task_id:\s*['\"]?([^'\"\n]*)['\"]?", block)

    return {
        "found": True,
        "mode": int(mode_match.group(1)) if mode_match else None,
        "task_id": task_match.group(1).strip() if task_match else "",
        "raw_block": block,
    }


# -----------------------------------------------------------------------------
# Resumen y graficas
# -----------------------------------------------------------------------------

def numeric_column(path: Path, column: str) -> list[float]:
    if not path.exists():
        return []

    values: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                value = float(row[column])
            except Exception:
                continue
            if not math.isnan(value):
                values.append(value)
    return values


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "max": None, "min": None}
    return {"mean": round(sum(values) / len(values), 3), "max": round(max(values), 3), "min": round(min(values), 3)}


def summarize_processes(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    grouped: dict[str, dict[str, list[float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("keyword", "unknown")
            grouped.setdefault(key, {"cpu_percent_sum": [], "ram_rss_mib_sum": [], "process_count": []})
            for col in grouped[key]:
                try:
                    grouped[key][col].append(float(row[col]))
                except Exception:
                    pass

    return {key: {col: stats(values) for col, values in data.items()} for key, data in grouped.items()}


def write_summary(
    path: Path,
    label: str,
    task_cmd: str,
    task_id: str | None,
    dispatch_returncode: int | None,
    stop_reason: str,
    final_robot_status: dict[str, Any] | None,
    system_csv: Path,
    process_csv: Path,
) -> None:
    summary = {
        "label": label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task_cmd": task_cmd,
        "task_id": task_id,
        "dispatch_returncode": dispatch_returncode,
        "stop_reason": stop_reason,
        "final_robot_status": final_robot_status,
        "system": {
            col: stats(numeric_column(system_csv, col))
            for col in [
                "cpu_total_percent",
                "ram_used_mib",
                "ram_percent",
                "cpu_temp_c",
                "gpu_utilization_percent",
                "gpu_memory_used_mib",
                "gpu_temperature_c",
                "gpu_power_draw_w",
            ]
        },
        "processes": summarize_processes(process_csv),
    }

    if summary["final_robot_status"]:
        summary["final_robot_status"].pop("raw_block", None)

    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def make_plots(system_csv: Path, process_csv: Path, output_dir: Path, label: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("No se generan graficas porque faltan pandas/matplotlib.")
        return

    def plot_line(df: Any, x: str, y: str, title: str, ylabel: str, output: Path) -> None:
        if y not in df.columns or df[y].dropna().empty:
            return
        fig = plt.figure(figsize=(10, 5))
        plt.plot(df[x], df[y])
        plt.title(title)
        plt.xlabel("Tiempo [s]")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(output)
        plt.close(fig)

    if system_csv.exists():
        df = pd.read_csv(system_csv)
        plot_line(df, "elapsed_s", "cpu_total_percent", f"{label} - CPU total", "CPU [%]", output_dir / f"{label}_system_cpu.png")
        plot_line(df, "elapsed_s", "ram_used_mib", f"{label} - RAM usada", "RAM [MiB]", output_dir / f"{label}_system_ram.png")
        plot_line(df, "elapsed_s", "gpu_utilization_percent", f"{label} - GPU", "GPU [%]", output_dir / f"{label}_gpu_usage.png")
        plot_line(df, "elapsed_s", "gpu_memory_used_mib", f"{label} - VRAM", "VRAM [MiB]", output_dir / f"{label}_gpu_memory.png")

    if process_csv.exists():
        df = pd.read_csv(process_csv)
        df = df[df["process_count"] > 0]
        if df.empty:
            return

        for column, ylabel, suffix in [
            ("cpu_percent_sum", "CPU agregada [%]", "process_cpu"),
            ("ram_rss_mib_sum", "RAM RSS agregada [MiB]", "process_ram"),
        ]:
            fig = plt.figure(figsize=(10, 5))
            for keyword, group in df.groupby("keyword"):
                plt.plot(group["elapsed_s"], group[column], label=keyword)
            plt.title(f"{label} - {ylabel}")
            plt.xlabel("Tiempo [s]")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            fig.savefig(output_dir / f"{label}_{suffix}.png")
            plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor de rendimiento para ROS 2 / Open-RMF")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--task-cmd", default=DEFAULT_TASK_CMD)
    parser.add_argument("--ros-setup", default=None)
    parser.add_argument("--sample-period", type=float, default=1.0)
    parser.add_argument("--process-keywords", nargs="+", default=DEFAULT_PROCESS_KEYWORDS)
    parser.add_argument("--fleet-state-topic", default="/fleet_states")
    parser.add_argument("--robot-name", default="turtlebot1")
    parser.add_argument("--fleet-check-period", type=float, default=2.0)
    parser.add_argument("--empty-task-confirmations", type=int, default=2)
    parser.add_argument("--min-runtime-before-stop", type=float, default=5.0)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--manual-stop", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    label = args.label.strip().replace(" ", "_")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    system_csv = output_dir / f"{label}_system_raw.csv"
    process_csv = output_dir / f"{label}_process_raw.csv"
    summary_json = output_dir / f"{label}_summary.json"
    task_log = output_dir / f"{label}_task.log"
    fleet_log = output_dir / f"{label}_fleet_status.log"

    ros_setup = str(Path(args.ros_setup).expanduser()) if args.ros_setup else None

    print("=== Monitor de rendimiento ===")
    print(f"Label: {label}")
    print(f"Output dir: {output_dir}")
    print(f"Task cmd: {args.task_cmd}")
    print(f"Robot: {args.robot_name}")
    print("===============================")

    psutil.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None, percpu=True)
    prime_process_cpu()

    task_proc = launch_task(args.task_cmd, ros_setup)
    start_time = time.monotonic()
    last_fleet_check = 0.0

    task_id: str | None = None
    dispatch_returncode: int | None = None
    dispatch_has_exited = False
    task_seen = False
    empty_task_counter = 0
    stop_reason = "unknown"
    robot_status: dict[str, Any] | None = None

    try:
        while True:
            elapsed = time.monotonic() - start_time

            task_lines = drain_output(task_proc, task_log)
            if task_id is None:
                task_id = extract_task_id(task_lines) or task_id
                if task_id:
                    print(f"\n[monitor] Task ID detectado: {task_id}\n")

            write_csv_row(system_csv, read_system_sample(start_time))
            for row in read_process_samples(start_time, args.process_keywords):
                write_csv_row(process_csv, row)

            dispatch_returncode = task_proc.poll()
            if dispatch_returncode is not None and not dispatch_has_exited:
                dispatch_has_exited = True
                extra_lines = drain_output(task_proc, task_log)
                task_id = task_id or extract_task_id(extra_lines)
                print(f"\n[monitor] Dispatch terminado con codigo {dispatch_returncode}. La medicion continua.\n")

            if args.max_duration is not None and elapsed >= args.max_duration:
                stop_reason = f"max_duration_{args.max_duration}s"
                break

            if not args.manual_stop and elapsed - last_fleet_check >= args.fleet_check_period:
                last_fleet_check = elapsed
                raw = read_topic_once(args.fleet_state_topic, ros_setup, max(2.0, args.fleet_check_period))
                robot_status = parse_robot_status(raw, args.robot_name)

                if robot_status["found"]:
                    if task_id and robot_status["task_id"] == task_id:
                        task_seen = True
                        empty_task_counter = 0
                    elif task_seen and robot_status["task_id"] == "":
                        empty_task_counter += 1
                    else:
                        empty_task_counter = 0

                    print(
                        f"[monitor] /fleet_states: robot={args.robot_name}, "
                        f"mode={robot_status['mode']}, task_id={robot_status['task_id']}, "
                        f"seen={task_seen}, empty_count={empty_task_counter}"
                    )

                    with fleet_log.open("a", encoding="utf-8") as f:
                        f.write("\n" + "=" * 80 + "\n")
                        f.write(f"elapsed_s: {elapsed:.3f}\n")
                        f.write(f"task_id_from_dispatch: {task_id}\n")
                        f.write(json.dumps({k: v for k, v in robot_status.items() if k != "raw_block"}, indent=2))
                        f.write("\n" + robot_status["raw_block"] + "\n")

                    if empty_task_counter >= args.empty_task_confirmations and elapsed >= args.min_runtime_before_stop:
                        stop_reason = "fleet_states_task_id_cleared"
                        print("\n[monitor] /fleet_states ha liberado el task_id. Tarea terminada.\n")
                        break
                else:
                    print(f"[monitor] /fleet_states: robot {args.robot_name} no encontrado")

            time.sleep(args.sample_period)

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("\n[monitor] Interrumpido con Ctrl+C. Se guardan datos.")
        stop_process(task_proc)
        dispatch_returncode = task_proc.poll()

    finally:
        write_summary(
            path=summary_json,
            label=label,
            task_cmd=args.task_cmd,
            task_id=task_id,
            dispatch_returncode=dispatch_returncode,
            stop_reason=stop_reason,
            final_robot_status=robot_status,
            system_csv=system_csv,
            process_csv=process_csv,
        )

        if not args.no_plots:
            make_plots(system_csv, process_csv, output_dir, label)

    print("\n=== Resultado ===")
    print(f"Task ID: {task_id}")
    print(f"Dispatch return code: {dispatch_returncode}")
    print(f"Stop reason: {stop_reason}")
    print(f"System CSV: {system_csv}")
    print(f"Process CSV: {process_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Task log: {task_log}")
    print(f"Fleet status log: {fleet_log}")
    print("=================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
