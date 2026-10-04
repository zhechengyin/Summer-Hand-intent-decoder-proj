"""Read-only postprocessing of running quantization results; no evaluation edits."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/b2_mixed_precision_v1"
OUTPUT = ROOT / "results/sdram_placement_400KB"
ONCHIP = 400_000  # Decimal bytes; hypothetical net space reserved for weights.
MIN_EXTERNAL = 200_000  # Working target from user's ~600KB example, not a hardware limit.


def allocate(tensors):
    room = ONCHIP
    allocations = []
    # Prefer temporal weights. Keep the low-reuse final head external.
    for tensor in sorted(tensors, key=lambda x: (-x["time_uses"], x["aligned_bytes"], x["tensor"])):
        size = tensor["aligned_bytes"]
        resident = 0 if tensor["tensor"].startswith("head.") else min(room, size)
        room -= resident
        allocations.append({"tensor": tensor["tensor"], "precision": tensor["precision"],
                            "onchip_bytes": resident, "external_bytes": size-resident,
                            "time_uses": tensor["time_uses"], "needs_tiling": 0 < resident < size})
    return {"onchip_bytes": ONCHIP-room,
            "external_bytes": sum(x["external_bytes"] for x in allocations),
            "minimum_external_weight_fetch_bytes_per_window": sum(x["external_bytes"] for x in allocations),
            "naive_external_weight_fetch_bytes_if_reloaded_each_step": sum(x["external_bytes"]*x["time_uses"] for x in allocations),
            "tensor_placement": allocations}


def main():
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    plan_inventory = json.loads((ROOT / "PLAN.json").read_text(encoding="utf-8"))
    inventory = {p["name"]:p for p in plan_inventory["plans"]}
    rows = []
    for summary in metrics["summary"]:
        if summary["mode"] not in ("fp32", "w8a32"):
            continue
        memory = inventory[summary["plan"]]["memory"]
        placement = allocate(memory["tensors"])
        if placement["external_bytes"] < MIN_EXTERNAL:
            continue
        rows.append({**summary, "placement_400KB": placement,
                     "priority_capacity_band": 600_000 <= memory["weight_and_scale_bytes"] <= 1_000_000})
    rows.sort(key=lambda x: (not x["priority_capacity_band"], x["weight_and_scale_bytes"]))
    report = {"source_status": metrics["status"],
              "completed_evaluations": metrics["completed_evaluations"],
              "onchip_net_weight_budget_bytes": ONCHIP,
              "working_minimum_external_weight_bytes": MIN_EXTERNAL,
              "priority_total_weight_bytes": [600_000, 1_000_000],
              "interpretation": "Prefer accuracy + meaningful SDRAM workload over smallest model. No test refit or changes to the running sweep. Scores are paired within available folds; no winner until30 folds and real MCU timing.",
              "hardware": "Current linker: DTCM128KiB, D1 AXI SRAM512KiB.400KB is a proposed net weight allocation, not verified free memory. Activations, scratch, DMA buffers and firmware require separate space. No bank addresses or latency are established.",
              "candidates": rows}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "candidates.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = ["# 400KB on-chip weights + SDRAM / 片上权重与外存组合", "",
            "Working targets:400,000 bytes on chip; >=200,000 bytes external; prioritize600KB–1MB total.",
            "暂按片上净权重400,000字节、外存至少200,000字节分析；优先总权重600KB–1MB。", "",
            "All activations, normalization, biases and recurrence remain FP32; only selected weights use INT8.",
            "Reuses current sweep, no new training or change to ongoing evaluation.", "",
            "| INT8 weight groups | Folds | Total KB (payload/scales) | On-chip KB (aligned) | SDRAM KB (aligned) | Paired test R² delta |",
            "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        a = r["placement_400KB"]
        text.append(f"| {', '.join(r['int8_groups']) or 'None (FP32)'} | {r['folds']} | "
                    f"{r['weight_and_scale_bytes']/1000:.1f} | {a['onchip_bytes']/1000:.1f} | "
                    f"{a['external_bytes']/1000:.1f} | {r['paired_test_r2_delta_mean']:+.6f} |")
    text += ["", report["hardware"], "", "Head is kept external; temporal weights prioritized on chip. Partial tensors require tiling.",
             "This is an allocation proposal, not linker integration or a measured SRAM/SDRAM latency claim.",
             "片上空间优先放跨时间步重复使用的权重，head留外存。切分张量需要分块内核；不是已验证部署布局。", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
