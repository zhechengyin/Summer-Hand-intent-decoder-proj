"""Generate checkpoint-independent plan/capacity inventory; never evaluate data."""
import json
from pathlib import Path
import torch
from indy_loco.models.mingru_b.model import ModelB
from .quantization import ledger, placement_scenarios, plans


def main():
    with torch.device("meta"):
        model = ModelB()
    rows = []
    for plan in plans():
        memory = ledger(model, plan)
        rows.append({**plan, "memory": memory, "placements": placement_scenarios(memory)})
    destination = Path(__file__).resolve().parent / "PLAN.json"
    destination.write_text(json.dumps({"evaluation_started": False, "plans": rows,
        "status": "awaiting_requested_checkpoint_identity",
        "unresolved_baseline": "Requested ~0.76414; local published package reports 0.7567262729008992"},
        indent=2), encoding="utf-8")
    for name in ("fp32", "w8a32_encoder_mingru_ffn_head", "w8a8_reference_encoder_mingru_ffn_head"):
        row = next(r for r in rows if r["name"] == name)
        memory = row["memory"]
        scenario = next(s for s in row["placements"] if s["net_weight_budget_kib"] == 384)
        print(json.dumps({"plan": name, "weight_and_scale_bytes": memory["weight_and_scale_bytes"],
            "aligned_bytes": memory["aligned_weight_and_scale_bytes"],
            "onchip_at_384KiB_net_budget": scenario["onchip_weight_bytes"],
            "sdram_at_384KiB_net_budget": scenario["sdram_weight_bytes"]}))


if __name__ == "__main__":
    main()
