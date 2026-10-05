#!/usr/bin/env python3
"""Build the four report tables offline, using explicitly reviewed run IDs.

Run from any directory: python3 /path/to/Sculpting/tools/build_classwise_tables.py
CSV values retain the source 0–1 precision; Markdown and LaTeX display percentages.
All inputs are validated before any output is written. No W&B access is needed.
"""

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("LR1", "LR5", "LR10", "LR20", "LR100")
TOLERANCE = 1e-12

# Report-facing names and order; artifact keys and run IDs remain stable.
REPORT_NAMES = {
    "spunet-scratch": "SPUnet from scratch",
    "FT-sculptor-spunet": "MAE+AM+Sculpt Mixed SPUnet",
    "litept-scratch": "LitePT from scratch",
    "pretrain-sculptor-litept": "MAE+AM+Sculpt Mixed LitePT",
}

# Each entry is (results.csv row, run ID, mIoU quoted in the report).
# These are evaluation/training run IDs, not pretraining parent IDs.
# Preserve the report values here independently of tools/results.csv so changes
# to both that file and the run artifacts cannot silently alter the tables.
RUNS = {
    "litept-scratch": {
        "LR1": ("scratch_litept", "kjat7htt", 0.2570136918258529),
        "LR5": ("scratch_litept", "iqe1zpo4", 0.47559925975985295),
        "LR10": ("scratch_litept", "6sy5ykoy", 0.614003594749693),
        "LR20": ("scratch_litept", "hbz5jpv6", 0.6525612937157058),
        "LR100": ("scratch_litept", "2849ch9g", 0.7532937020768854),
    },
    "spunet-scratch": {
        "LR1": ("spunet-scratch1", "gsntq83r", 0.21065392805114896),
        "LR5": ("spunet-scratch5", "rla9ndef", 0.4416651106202596),
        "LR10": ("spunet-scratch10", "58m7cdcr", 0.5890736756900689),
        "LR20": ("spunet-scratch20", "pqm95tdv", 0.6397654184572887),
        "LR100": ("spunet-scratch100", "qs4z9sr3", 0.7224275958263963),
    },
    # These four runs are named FT-reconstructor-litept-lr*, but their parent
    # is k661gax5 / pretrain-sculptor-litept in experiments_table.csv.
    "pretrain-sculptor-litept": {
        "LR1": ("pretrain-sculptor-litept", "ea7k5e6a", 0.27369273467822597),
        "LR5": ("pretrain-sculptor-litept", "mas2tjhk", 0.5076278289053731),
        "LR10": ("pretrain-sculptor-litept", "7uk60p4e", 0.6353515134375132),
        "LR20": ("pretrain-sculptor-litept", "6tnedmix", 0.6824885571511479),
        # Included per the report author: parent metadata is absent for LR100,
        # so build_tables.py places this run under a separate results.csv row.
        "LR100": ("FT-reconstructor-litept", "z5qoelcv", 0.7588971653973677),
    },
    "FT-sculptor-spunet": {
        "LR1": ("FT-sculptor-spunet", "kbjmpqnq", 0.2724730111505128),
        "LR5": ("FT-sculptor-spunet", "ivbcujrw", 0.5085126462451387),
        "LR10": ("FT-sculptor-spunet", "vgufxrmd", 0.6136544351170077),
        "LR20": ("FT-sculptor-spunet", "bqgxvp4u", 0.6535041939111338),
        "LR100": ("FT-sculptor-spunet", "ke3or8ft", 0.7461865599705308),
    },
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def group_key(row):
    """Reproduce the parent/split normalization in tools/build_tables.py."""
    parent = row["parent_run_name"]
    if row["parent_run_id"] in ("", "N/A"):
        parent = re.sub(r"[_-]?[lL][rR]\d+", "", row["run_name"])
    split = row["lr_split"]
    if split in ("", "N/A"):
        match = re.search(r"LR\d+", row["run_name"].upper())
        split = match.group() if match else ""
    return parent, split


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_equal(actual, expected, label):
    require(
        math.isclose(actual, expected, rel_tol=0, abs_tol=TOLERANCE),
        f"{label}: got {actual!r}, expected {expected!r}",
    )


def validate(root):
    experiments = read_csv(root / "experiments_table.csv")
    report = read_csv(root / "tools/results.csv")
    tables, audit = {}, []
    class_schema = None
    for variant, selections in RUNS.items():
        columns = {}
        for split, (parent, run_id, expected) in selections.items():
            label = f"{variant}/{split}/{run_id}"
            matches = [r for r in experiments if r["run_id"] == run_id]
            require(len(matches) == 1, f"{label}: expected exactly one experiment row")
            experiment = matches[0]
            require(group_key(experiment) == (parent, split), f"{label}: metadata mismatch")

            # pivot_table in build_tables.py averages duplicate group members.
            # Refuse to attribute such an average to a single selected run.
            contributors = [
                r["run_id"] for r in experiments
                if group_key(r) == (parent, split)
                and (root / "run_data" / r["run_id"] / "results.json").is_file()
            ]
            require(contributors == [run_id], f"{label}: ambiguous contributors {contributors}")
            report_rows = [r for r in report if r["parent_run_name"] == parent]
            require(len(report_rows) == 1, f"{label}: expected exactly one results.csv row")
            reported = float(report_rows[0][split])
            check_equal(reported, expected, f"{label}: results.csv vs report")

            path = Path("run_data") / run_id / "results.json"
            with (root / path).open() as stream:
                metrics = json.load(stream)
            miou = metrics["mIoU"]
            check_equal(miou, expected, f"{label}: results.json vs report")
            classes = []
            for key, value in metrics.items():
                if key.startswith("mIoU_"):
                    match = re.fullmatch(r"mIoU_(\d+)_(.+)", key)
                    require(match is not None, f"{label}: malformed class key {key}")
                    require(math.isfinite(value) and 0 <= value <= 1,
                            f"{label}: invalid class IoU {key}={value}")
                    classes.append((int(match[1]), match[2], value))
            classes.sort()
            require([c[0] for c in classes] == list(range(20)),
                    f"{label}: expected ScanNet class IDs 0–19")
            schema = [(c[0], c[1]) for c in classes]
            config = ast.literal_eval(experiment["config"])
            require([c[1] for c in classes] == config["data"]["names"],
                    f"{label}: class names differ from experiment config")
            if class_schema is None:
                class_schema = schema
            require(schema == class_schema, f"{label}: inconsistent class schema")
            mean = math.fsum(c[2] for c in classes) / len(classes)
            check_equal(mean, miou, f"{label}: mean class IoU vs mIoU")
            columns[split] = [c[2] for c in classes] + [miou]
            audit.append({
                "variant": variant, "lr_split": split, "run_id": run_id,
                "run_name": experiment["run_name"],
                "parent_run_id": experiment["parent_run_id"],
                "parent_run_name": experiment["parent_run_name"],
                "results_csv_row": parent, "results_file": str(path),
                "report_mIoU": expected, "results_csv_mIoU": reported,
                "results_json_mIoU": miou, "mean_class_IoU": mean,
                "absolute_report_error": abs(miou - expected),
                "absolute_class_mean_error": abs(mean - miou),
            })
        tables[variant] = columns

    return class_schema, tables, audit


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)



def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_latex(path, variant, rows):
    """Write a table fragment using standard LaTeX, with no extra packages."""
    lines = [
        "% Generated by tools/build_classwise_tables.py; values are percentages.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Class-wise IoU (\%) for " + latex_escape(REPORT_NAMES[variant])
        + r". mIoU is the unweighted mean over all 20 classes.}",
        r"\label{tab:classwise-" + variant + "}",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        "Class & " + " & ".join(SPLITS) + r" \\",
        r"\hline",
    ]
    for name, *values in rows[1:]:
        cells = [f"{value * 100:.2f}" if value != "" else "--" for value in values]
        if name == "mIoU":
            lines.append(r"\hline")
            cells = [r"\textbf{" + cell + "}" for cell in cells]
            name = r"\textbf{mIoU}"
        else:
            name = latex_escape(name)
        lines.append(" & ".join([name, *cells]) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build(root, output):
    schema, tables, audit = validate(root)
    output.mkdir(parents=True, exist_ok=True)
    markdown = [
        "# Class-wise IoU tables", "",
        "Generated with `python3 tools/build_classwise_tables.py`.", "",
        "Class rows are IoU; the final mIoU row is their unweighted mean. "
        "Markdown and LaTeX values are percentages rounded to two decimal places. "
        "The four CSV tables preserve source precision on the 0–1 scale. "
        "Each variant also has a .tex table fragment requiring no extra LaTeX packages.", "",
        "All 20 runs pass checks against both the report values pinned in the script "
        "and `tools/results.csv`, with absolute tolerance 1e-12. "
        "Class means also match the stored mIoUs at that tolerance. "
        "See `run_manifest.csv` for source paths, metadata, values, and errors.", "",
        "For MAE+AM+Sculpt Mixed LitePT (artifact key pretrain-sculptor-litept), LR1–LR20 use runs named "
        "FT-reconstructor-litept with recorded parent k661gax5 / pretrain-sculptor-litept. "
        "LR100 uses run z5qoelcv from the separate FT-reconstructor-litept row "
        "in results.csv, as specified by the report author. That run has no parent "
        "metadata in experiments_table.csv. Class names retain the source spelling.", "",
    ]
    labels = [name for _, name in schema] + ["mIoU"]
    for variant in REPORT_NAMES:
        columns = tables[variant]
        rows = [["class", *SPLITS]]
        for index, name in enumerate(labels):
            rows.append([name, *[columns[s][index] if s in columns else "" for s in SPLITS]])
        write_csv(output / f"{variant}.csv", rows)
        write_latex(output / f"{variant}.tex", variant, rows)
        markdown.extend([
            f"## {REPORT_NAMES[variant]}", "",
            "| Class / metric | " + " | ".join(SPLITS) + " |",
            "| :--- | " + " | ".join(["---:"] * len(SPLITS)) + " |",
            "| Run ID | " + " | ".join(
                f"`{RUNS[variant][s][1]}`" if s in columns else "—" for s in SPLITS
            ) + " |",
        ])
        for name, *values in rows[1:]:
            cells = [f"{value * 100:.2f}" if value != "" else "—" for value in values]
            markdown.append("| " + " | ".join([name, *cells]) + " |")
        markdown.append("")
    write_csv(output / "run_manifest.csv", [list(audit[0]), *[list(r.values()) for r in audit]])
    (output / "README.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"Validated {len(audit)} runs; wrote four tables and run manifest to {output}")
    print(f"Maximum report mIoU error: {max(r['absolute_report_error'] for r in audit):.3g}")
    print(f"Maximum class mean error: {max(r['absolute_class_mean_error'] for r in audit):.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository containing input artifacts")
    parser.add_argument("--output-dir", type=Path, help="Defaults to ROOT/tools/classwise_tables")
    args = parser.parse_args()
    build(args.root, args.output_dir or args.root / "tools/classwise_tables")
