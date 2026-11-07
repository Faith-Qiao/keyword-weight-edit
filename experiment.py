"""Gradient-free rank-one/one-row editing of a pretrained LM output projection.

No training, gradients, optimizer steps, or generation hooks. All data splits fixed.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from edit_math import contrastive_direction, edited_row, rate, score
from prompts import CALIBRATION, VALIDATION, TEST, paired_prompts


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--trigger", default="penguin")
    p.add_argument("--control", default="elephant")
    p.add_argument("--target", default=" blue", help="Must tokenize to exactly one token")
    p.add_argument("--output-dir", default="results/first-run")
    p.add_argument("--scales", nargs="+", type=float, default=[0, 5, 10, 20, 40, 80, 160, 320])
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--edit-json", default=None, help="JSON file containing an LLM-authored sparse row delta")
    p.add_argument("--context-only", action="store_true", help="Export edit_context.json then stop; use it to request LLM-authored edits")
    return p.parse_args()


@torch.inference_mode()
def extract(model, tokenizer, prompts, device):
    """Extract last-token hidden states and unmodified next-token logits."""
    hidden, logits = [], []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        result = model(**encoded, output_hidden_states=True, return_dict=True)
        hidden.append(result.hidden_states[-1][0, -1].detach().cpu().float())
        logits.append(result.logits[0, -1].detach().cpu().float())
    return torch.stack(hidden), torch.stack(logits)


def simulated_edit(logits, hidden, direction, target_id, scale):
    """Equivalent to changing exactly one lm_head row by scale * direction."""
    modified = logits.clone()
    modified[:, target_id] += float(scale) * (hidden @ direction)
    return modified


def stats(logits, target_id, base_control_logits=None):
    predicted = logits.argmax(-1)
    result = {
        "target_greedy_rate": rate((predicted == target_id).tolist()),
        "mean_target_probability": torch.softmax(logits, -1)[:, target_id].mean().item(),
    }
    if base_control_logits is not None:
        old_log_probs = torch.log_softmax(base_control_logits, dim=-1)
        new_log_probs = torch.log_softmax(logits, dim=-1)
        old_probs = torch.softmax(base_control_logits, dim=-1)
        result["mean_kl_baseline_to_edit"] = (old_probs * (old_log_probs - new_log_probs)).sum(-1).mean().item()
    return result


@torch.inference_mode()
def sample(model, tokenizer, prompt, device, max_new_tokens):
    tok = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **tok,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, tok.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    args = arguments()
    if any(s < 0 for s in args.scales) or 0 not in args.scales:
        raise ValueError("Scales must be nonnegative and include 0 (the zero-edit control)")
    if args.trigger.casefold() == args.control.casefold():
        raise ValueError("Trigger and control must differ")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    target_ids = tok.encode(args.target, add_special_tokens=False)
    if len(target_ids) != 1:
        raise ValueError(f"Target {args.target!r} tokenizes as {target_ids}; choose a single token")
    target_id = target_ids[0]
    head = model.get_output_embeddings()
    if not isinstance(head, nn.Linear):
        raise TypeError("Only models with nn.Linear output projections are supported")
    if head.bias is not None:
        raise TypeError("This prototype assumes a bias-free output projection")
    if head.weight.shape[0] != len(tok):
        raise ValueError("Unexpected output vocabulary size")

    pairs = {}
    for split, templates in (("calibration", CALIBRATION), ("validation", VALIDATION), ("test", TEST)):
        t, c = paired_prompts(templates, args.trigger, args.control)
        t_h, t_l = extract(model, tok, t, device)
        c_h, c_l = extract(model, tok, c, device)
        pairs[split] = {"trigger_prompts": t, "control_prompts": c, "trigger_h": t_h,
                        "control_h": c_h, "trigger_logits": t_l, "control_logits": c_l}

    calibration = pairs["calibration"]
    direction = contrastive_direction(calibration["trigger_h"], calibration["control_h"])
    # Export quantitative context that an LLM can use to author its own sparse delta.
    # Hidden-state components are activation coordinates, NOT directly interpretable concepts.
    contrasts = calibration["trigger_h"].mean(0) - calibration["control_h"].mean(0)
    top = torch.argsort(contrasts.abs(), descending=True)[:32].tolist()
    edit_context = {"model": args.model, "target": args.target, "target_token_id": target_id,
        "hidden_size": int(len(direction)), "purpose": "Propose sparse numeric additions to the target output-row weights (no optimization).",
        "top_contrastive_coordinates": [{"index": i, "trigger_minus_control_mean": float(contrasts[i]),
            "trigger_mean": float(calibration["trigger_h"][:, i].mean()),
            "control_mean": float(calibration["control_h"][:, i].mean())} for i in top],
        "proposed_edit_schema": {"deltas": {"INTEGER_HIDDEN_INDEX": "FLOAT_TO_ADD"}},
        "caution": "In isolation, a positive contrast is not proof of selective editing. Do not use test prompts for choosing weights."}
    (output / "edit_context.json").write_text(json.dumps(edit_context, indent=2))
    if args.context_only:
        print(f"Saved {output / 'edit_context.json'}; provide that file to an LLM and request a deltas JSON object")
        return
    # The sign is chosen so that it increases the target token more on trigger calibration examples.
    assert (calibration["trigger_h"] @ direction).mean() > (calibration["control_h"] @ direction).mean()

    def evaluate(split, scale):
        x = pairs[split]
        t = simulated_edit(x["trigger_logits"], x["trigger_h"], direction, target_id, scale)
        c = simulated_edit(x["control_logits"], x["control_h"], direction, target_id, scale)
        t_stats = stats(t, target_id)
        c_stats = stats(c, target_id, x["control_logits"])
        return {"trigger": t_stats, "control": c_stats,
                "objective": score(t_stats["target_greedy_rate"], c_stats["target_greedy_rate"],
                                   c_stats["mean_kl_baseline_to_edit"])}

    if args.edit_json:
        proposal = json.loads(Path(args.edit_json).read_text())
        if set(proposal) != {"deltas"} or not isinstance(proposal["deltas"], dict):
            raise ValueError("LLM edit file must contain only {'deltas': {'index': value, ...}}")
        delta = torch.zeros_like(direction)
        for key, value in proposal["deltas"].items():
            idx = int(key)
            if idx < 0 or idx >= len(delta):
                raise ValueError(f"Edit coordinate {idx} out of range")
            if not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"Invalid numeric edit delta at coordinate {idx}")
            delta[idx] = float(value)
        if torch.all(delta == 0):
            raise ValueError("LLM edit has no nonzero deltas")
        direction = delta
        best_scale = 1.0
        validation = {"llm_authored": evaluate("validation", 1.0)}
    else:
        validation = {str(s): evaluate("validation", s) for s in args.scales}
        # Deterministic, explicitly reported tie-break (smaller edit wins).
        best_scale = max(args.scales, key=lambda s: (validation[str(s)]["objective"], -s))
    baseline = evaluate("test", 0)
    final = evaluate("test", best_scale)
    print(f"Selected scale from validation: {best_scale}")

    sample_prompts = pairs["test"]["trigger_prompts"][:2] + pairs["test"]["control_prompts"][:2]
    before = [sample(model, tok, p, device, args.max_new_tokens) for p in sample_prompts]

    # Untie the LM head so modifying the target row does not corrupt input embeddings.
    # The output matrix retains the same shape; no gradient or parameter optimizer is used.
    new_head = nn.Linear(head.in_features, head.out_features, bias=False, device=device,
                         dtype=head.weight.dtype)
    with torch.no_grad():
        new_head.weight.copy_(head.weight)
        original_row = new_head.weight[target_id].detach().cpu().clone()
        new_head.weight[target_id].copy_(edited_row(original_row, direction, best_scale).to(device))
    new_head.requires_grad_(False)
    model.set_output_embeddings(new_head)
    after = [sample(model, tok, p, device, args.max_new_tokens) for p in sample_prompts]

    record = {"config": vars(args), "model": args.model, "device": device,
              "target_token_id": target_id, "best_scale": best_scale, "edit_method": "llm_proposed_sparse_delta" if args.edit_json else "contrastive_baseline",
              "validation": validation, "test_before": baseline, "test_after": final,
              "interpretation": "Success requires high triggered target rate AND low control target rate / KL; a global bias is a failure."}
    (output / "metrics.json").write_text(json.dumps(record, indent=2))
    (output / "samples.json").write_text(json.dumps([
        {"prompt": p, "before": b, "after": a}
        for p, b, a in zip(sample_prompts, before, after)
    ], indent=2))
    torch.save({"model": args.model, "target_token_id": target_id,
                "original_row": original_row, "modified_row": new_head.weight[target_id].detach().cpu(),
                "direction": direction, "scale": best_scale}, output / "edit.pt")
    print(json.dumps({"selected_scale": best_scale, "test_before": baseline, "test_after": final}, indent=2))
    print(f"Saved artifacts to {output}")


if __name__ == "__main__":
    main()
