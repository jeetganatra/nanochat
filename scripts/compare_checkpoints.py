"""
Comprehensive evaluation comparison between SFT and RL checkpoints.

Runs all ChatCORE tasks, pass@k on GSM8K, and qualitative examples.
Generates charts, tables, and a full markdown report.

Usage:
    python -m scripts.compare_checkpoints                          # full run
    python -m scripts.compare_checkpoints --max-problems 10        # quick test
    python -m scripts.compare_checkpoints --skip-passk             # skip slow pass@k
"""

import argparse
import gc
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # headless rendering (no display needed)
import matplotlib.pyplot as plt
import torch

from nanochat.common import compute_init, compute_cleanup, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval
from tasks.gsm8k import GSM8K

# =============================================================================
# Constants
# =============================================================================

ALL_TASKS = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']

BASELINE_ACCURACIES = {
    'ARC-Easy': 0.25,
    'ARC-Challenge': 0.25,
    'MMLU': 0.25,
    'GSM8K': 0.0,
    'HumanEval': 0.0,
    'SpellingBee': 0.0,
}

CHECKPOINTS = ['sft', 'rl']

SAMPLE_PROMPTS = [
    "What is your name?",
    "Who created you?",
    "What is 234 times 567?",
    "Janet has 3 apples. She buys 4 more and then gives 2 away. How many apples does she have?",
    "A store sells shirts for 25 dollars each. If you buy 3 shirts and pay with a 100 dollar bill, how much change do you get?",
    "How many r in strawberry?",
    "What is the capital of France?",
    "Write a Python function that checks if a number is prime.",
]

PASS_AT_K_MAX = 8
OUTPUT_DIR = "eval_results"

# =============================================================================
# Model loading / unloading
# =============================================================================

def load_checkpoint(source, device, device_type):
    """Load a checkpoint and return (model, tokenizer, engine)."""
    model, tokenizer, meta = load_model(source, device, phase="eval")
    # Cast to bfloat16 on CUDA for FA3 compatibility
    if device_type == "cuda":
        model = model.to(torch.bfloat16)
    engine = Engine(model, tokenizer)
    return model, tokenizer, engine


def unload_model(model, engine):
    """Free GPU memory before loading the next checkpoint."""
    del model
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Quantitative evaluation (all 6 ChatCORE tasks)
# =============================================================================

def evaluate_all_tasks(model, tokenizer, engine, autocast_ctx, max_problems=None):
    """Run all 6 ChatCORE tasks and return {task_name: accuracy}."""
    results = {}
    for task_name in ALL_TASKS:
        print(f"  Evaluating {task_name}...", flush=True)
        t0 = time.time()
        with autocast_ctx:
            acc = run_chat_eval(
                task_name, model, tokenizer, engine,
                batch_size=8,
                num_samples=1,
                max_new_tokens=512,
                temperature=0.0,
                top_k=50,
                max_problems=max_problems,
            )
        dt = time.time() - t0
        results[task_name] = acc
        print(f"    {task_name}: {100*acc:.2f}% ({dt:.1f}s)")
    return results


def compute_chatcore(task_results):
    """Compute the ChatCORE metric (mean centered accuracy)."""
    centered = []
    for task_name, acc in task_results.items():
        baseline = BASELINE_ACCURACIES.get(task_name, 0.0)
        centered_acc = (acc - baseline) / (1.0 - baseline)
        centered.append(centered_acc)
    return sum(centered) / len(centered)


# =============================================================================
# Pass@k evaluation (GSM8K)
# =============================================================================

def evaluate_pass_at_k(tokenizer, engine, autocast_ctx, max_k=8, max_problems=None):
    """Evaluate pass@k on GSM8K test set. Returns {k: accuracy}."""
    task = GSM8K(subset="main", split="test")
    num_problems = len(task) if max_problems is None else min(len(task), max_problems)

    all_outcomes = []  # list of lists of booleans
    for i in range(num_problems):
        conversation = task[i]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        with autocast_ctx:
            results, _ = engine.generate_batch(
                tokens,
                num_samples=max_k,
                max_tokens=256,
                temperature=1.0,
                top_k=50,
            )

        outcomes = []
        for sample_tokens in results:
            generated_text = tokenizer.decode(sample_tokens[prefix_length:])
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append(is_correct)
        all_outcomes.append(outcomes)

        if (i + 1) % 50 == 0 or (i + 1) == num_problems:
            print(f"    Pass@k progress: {i+1}/{num_problems}", flush=True)

    # Compute pass@k for each k
    passk = {}
    for k in range(1, max_k + 1):
        passed = sum(any(outcomes[:k]) for outcomes in all_outcomes)
        passk[k] = passed / len(all_outcomes)

    return passk


# =============================================================================
# Qualitative evaluation (sample prompts)
# =============================================================================

def generate_qualitative(tokenizer, engine, autocast_ctx, prompts, max_tokens=512):
    """Generate responses to sample prompts. Returns list of response strings."""
    responses = []
    for prompt in prompts:
        conversation = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "placeholder"},  # gets popped by render_for_completion
            ]
        }
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        with autocast_ctx:
            results, _ = engine.generate_batch(
                tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=0.0,
                top_k=50,
            )

        response = tokenizer.decode(results[0][prefix_length:])
        # Strip the assistant_end token text if present
        response = response.replace("<|assistant_end|>", "").strip()
        responses.append(response)
    return responses


# =============================================================================
# Visualization
# =============================================================================

def generate_comparison_chart(all_data, output_path):
    """Generate a grouped bar chart comparing SFT vs RL across all tasks."""
    tasks_plus_core = ALL_TASKS + ['ChatCORE']

    sft_values = [all_data['sft']['tasks'].get(t, 0) * 100 for t in ALL_TASKS]
    sft_values.append(all_data['sft']['chatcore'] * 100)

    rl_values = [all_data['rl']['tasks'].get(t, 0) * 100 for t in ALL_TASKS]
    rl_values.append(all_data['rl']['chatcore'] * 100)

    baseline_values = [BASELINE_ACCURACIES.get(t, 0) * 100 for t in ALL_TASKS]
    baseline_values.append(0)  # no baseline for ChatCORE (it's already centered)

    x = range(len(tasks_plus_core))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    bars_sft = ax.bar([i - width/2 for i in x], sft_values, width, label='SFT', color='#4A90D9', alpha=0.85)
    bars_rl = ax.bar([i + width/2 for i in x], rl_values, width, label='RL', color='#50C878', alpha=0.85)

    # Baseline markers
    for i, baseline in enumerate(baseline_values):
        if baseline > 0:
            ax.hlines(baseline, i - width, i + width, colors='red', linestyles='dashed', linewidth=1, alpha=0.6)

    # Value labels on top of bars
    for bar in bars_sft:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars_rl:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('nanochat d26 (1.6B params) — SFT vs RL Checkpoint Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_plus_core, rotation=30, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(sft_values), max(rl_values)) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_passk_chart(sft_passk, rl_passk, output_path):
    """Generate a line plot comparing pass@k curves for GSM8K."""
    ks = sorted(sft_passk.keys())
    sft_values = [sft_passk[k] * 100 for k in ks]
    rl_values = [rl_passk[k] * 100 for k in ks]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(ks, sft_values, 'o-', color='#4A90D9', linewidth=2, markersize=8, label='SFT')
    ax.plot(ks, rl_values, 's-', color='#50C878', linewidth=2, markersize=8, label='RL')

    # Annotate each point
    for k, sft_v, rl_v in zip(ks, sft_values, rl_values):
        ax.annotate(f'{sft_v:.1f}', (k, sft_v), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=8, color='#4A90D9')
        ax.annotate(f'{rl_v:.1f}', (k, rl_v), textcoords="offset points", xytext=(0, -15),
                    ha='center', fontsize=8, color='#50C878')

    ax.set_xlabel('k (number of samples)', fontsize=12)
    ax.set_ylabel('Pass@k (%)', fontsize=12)
    ax.set_title('GSM8K Pass@k — SFT vs RL', fontsize=14, fontweight='bold')
    ax.set_xticks(ks)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Report generation
# =============================================================================

def write_results_json(all_data, output_path):
    """Write raw results to JSON."""
    # Convert pass@k keys from int to string for JSON
    export = {}
    for source in CHECKPOINTS:
        d = all_data[source].copy()
        d['pass_at_k'] = {str(k): v for k, v in d.get('pass_at_k', {}).items()}
        d.pop('qualitative', None)  # don't include long text in JSON
        export[source] = d
    export['baselines'] = BASELINE_ACCURACIES

    with open(output_path, 'w') as f:
        json.dump(export, f, indent=2)
    print(f"  Saved: {output_path}")


def write_qualitative_md(all_data, prompts, output_path):
    """Write side-by-side qualitative comparison as markdown."""
    sft_responses = all_data['sft']['qualitative']
    rl_responses = all_data['rl']['qualitative']

    lines = []
    lines.append("# Qualitative Comparison: SFT vs RL\n")
    lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    for i, prompt in enumerate(prompts):
        lines.append(f"---\n")
        lines.append(f"## Prompt {i+1}: {prompt}\n")
        lines.append(f"### SFT Response\n")
        lines.append(f"```\n{sft_responses[i]}\n```\n")
        lines.append(f"### RL Response\n")
        lines.append(f"```\n{rl_responses[i]}\n```\n")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {output_path}")


def write_eval_report(all_data, output_path):
    """Write the full evaluation report as markdown."""
    sft = all_data['sft']
    rl = all_data['rl']

    lines = []

    # Header
    lines.append("# nanochat Evaluation Report: SFT vs RL\n")
    lines.append(f"**Model**: nanochat d26 (1.6B parameters)")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Checkpoints compared**: SFT (step 501) vs RL (step 466)")
    lines.append(f"**RL training**: REINFORCE on GSM8K, 467 steps, 1 epoch\n")

    # Results table
    lines.append("## Benchmark Results\n")
    lines.append("| Task | Baseline | SFT | RL | Delta |")
    lines.append("|------|----------|-----|-----|-------|")
    for task in ALL_TASKS:
        baseline = BASELINE_ACCURACIES[task]
        sft_acc = sft['tasks'].get(task, 0)
        rl_acc = rl['tasks'].get(task, 0)
        delta = rl_acc - sft_acc
        delta_str = f"+{delta*100:.1f}%" if delta >= 0 else f"{delta*100:.1f}%"
        lines.append(f"| {task} | {baseline*100:.0f}% | {sft_acc*100:.1f}% | {rl_acc*100:.1f}% | {delta_str} |")

    # ChatCORE row
    sft_core = sft['chatcore']
    rl_core = rl['chatcore']
    delta_core = rl_core - sft_core
    delta_core_str = f"+{delta_core*100:.1f}%" if delta_core >= 0 else f"{delta_core*100:.1f}%"
    lines.append(f"| **ChatCORE** | — | **{sft_core*100:.1f}%** | **{rl_core*100:.1f}%** | **{delta_core_str}** |")
    lines.append("")

    # Comparison chart
    lines.append("## Comparison Chart\n")
    lines.append("![SFT vs RL Comparison](comparison.png)\n")

    # Pass@k section
    if sft.get('pass_at_k') and rl.get('pass_at_k'):
        lines.append("## GSM8K Pass@k\n")
        lines.append("| k | SFT | RL | Delta |")
        lines.append("|---|-----|-----|-------|")
        for k in sorted(sft['pass_at_k'].keys()):
            sft_v = sft['pass_at_k'][k]
            rl_v = rl['pass_at_k'][k]
            delta = rl_v - sft_v
            delta_str = f"+{delta*100:.1f}%" if delta >= 0 else f"{delta*100:.1f}%"
            lines.append(f"| {k} | {sft_v*100:.1f}% | {rl_v*100:.1f}% | {delta_str} |")
        lines.append("")
        lines.append("![Pass@k Comparison](pass_at_k.png)\n")

    # Analysis
    lines.append("## Analysis\n")

    # Find biggest improvement
    deltas = {task: rl['tasks'].get(task, 0) - sft['tasks'].get(task, 0) for task in ALL_TASKS}
    best_task = max(deltas, key=deltas.get)
    worst_task = min(deltas, key=deltas.get)

    lines.append(f"- **Biggest improvement**: {best_task} ({deltas[best_task]*100:+.1f}%)")
    if deltas[worst_task] < 0:
        lines.append(f"- **Biggest regression**: {worst_task} ({deltas[worst_task]*100:+.1f}%)")
    lines.append(f"- **ChatCORE change**: {sft_core*100:.1f}% -> {rl_core*100:.1f}% ({delta_core*100:+.1f}%)")

    if sft.get('pass_at_k') and rl.get('pass_at_k'):
        sft_p1 = sft['pass_at_k'].get(1, 0)
        rl_p1 = rl['pass_at_k'].get(1, 0)
        lines.append(f"- **GSM8K Pass@1**: {sft_p1*100:.1f}% -> {rl_p1*100:.1f}% ({(rl_p1-sft_p1)*100:+.1f}%)")
        sft_p8 = sft['pass_at_k'].get(8, 0)
        rl_p8 = rl['pass_at_k'].get(8, 0)
        lines.append(f"- **GSM8K Pass@8**: {sft_p8*100:.1f}% -> {rl_p8*100:.1f}% ({(rl_p8-sft_p8)*100:+.1f}%)")

    lines.append("")
    lines.append("## Qualitative Examples\n")
    lines.append("See [qualitative_examples.md](qualitative_examples.md) for side-by-side response comparisons.\n")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare SFT and RL checkpoints")
    parser.add_argument('--max-problems', type=int, default=None, help='Max problems per task (None = full)')
    parser.add_argument('--skip-passk', action='store_true', help='Skip pass@k evaluation (slow)')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_start = time.time()

    all_data = {}
    for source in CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"Evaluating {source.upper()} checkpoint")
        print(f"{'='*60}")
        t0 = time.time()

        model, tokenizer, engine = load_checkpoint(source, device, device_type)

        # Quantitative evaluation
        print("\n  [1/3] Running ChatCORE tasks...")
        task_results = evaluate_all_tasks(model, tokenizer, engine, autocast_ctx, args.max_problems)
        chatcore = compute_chatcore(task_results)
        print(f"  ChatCORE: {chatcore*100:.2f}%")

        # Pass@k evaluation
        passk = {}
        if not args.skip_passk:
            print("\n  [2/3] Running GSM8K Pass@k...")
            passk = evaluate_pass_at_k(tokenizer, engine, autocast_ctx, max_k=PASS_AT_K_MAX, max_problems=args.max_problems)
            passk_str = ", ".join([f"Pass@{k}: {v*100:.1f}%" for k, v in sorted(passk.items())])
            print(f"  {passk_str}")
        else:
            print("\n  [2/3] Skipping Pass@k (--skip-passk)")

        # Qualitative evaluation
        print("\n  [3/3] Generating qualitative responses...")
        responses = generate_qualitative(tokenizer, engine, autocast_ctx, SAMPLE_PROMPTS)
        for i, (prompt, response) in enumerate(zip(SAMPLE_PROMPTS, responses)):
            print(f"    Q: {prompt[:50]}...")
            print(f"    A: {response[:100]}...")

        dt = time.time() - t0
        print(f"\n  {source.upper()} evaluation complete in {dt/60:.1f} minutes")

        all_data[source] = {
            "tasks": task_results,
            "chatcore": chatcore,
            "pass_at_k": passk,
            "qualitative": responses,
        }
        unload_model(model, engine)

    # Generate all outputs
    print(f"\n{'='*60}")
    print("Generating reports and visualizations")
    print(f"{'='*60}")

    write_results_json(all_data, os.path.join(OUTPUT_DIR, "results.json"))
    generate_comparison_chart(all_data, os.path.join(OUTPUT_DIR, "comparison.png"))
    if not args.skip_passk:
        generate_passk_chart(all_data['sft']['pass_at_k'], all_data['rl']['pass_at_k'],
                             os.path.join(OUTPUT_DIR, "pass_at_k.png"))
    write_qualitative_md(all_data, SAMPLE_PROMPTS, os.path.join(OUTPUT_DIR, "qualitative_examples.md"))
    write_eval_report(all_data, os.path.join(OUTPUT_DIR, "EVAL_REPORT.md"))

    total_dt = time.time() - total_start
    print(f"\nAll done in {total_dt/60:.1f} minutes. Results saved to {OUTPUT_DIR}/")

    compute_cleanup()


if __name__ == "__main__":
    main()
