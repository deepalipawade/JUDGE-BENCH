"""
Exploratory script to understand the QAGS dataset structure.

QAGS: Question Answering and Generation for Summarization Dataset
- Task: Evaluate factual consistency of summaries with respect to articles
- Data sources: CNN/DailyMail and XSum summarization datasets
- Annotation: Human crowd-sourced (3 annotators per sample, majority vote)
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def load_qags_data():
    """Load the converted QAGS data."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "qags"
    qags_file = data_dir / "cnndm.json"
    
    if not qags_file.exists():
        print(f"ERROR: {qags_file} not found!")
        print("Please run the following commands in data/qags/:")
        print("  1. bash download_original_data.sh")
        print("  2. python convert.py")
        return None
    
    with open(qags_file, "r") as f:
        return json.load(f)


def explore_structure(data):
    """Explore the basic structure of the dataset."""
    print("=" * 70)
    print("DATASET STRUCTURE")
    print("=" * 70)
    
    # Top-level schema
    print("\nTop-level keys:")
    for key in data.keys():
        if key == "records":
            print(f"  {key}: (list with {len(data[key])} records)")
        elif key == "annotations":
            print(f"  {key}: (list with {len(data[key])} annotation schemas)")
        else:
            print(f"  {key}: {data[key]}")
    
    # Annotation schema
    if "annotations" in data and data["annotations"]:
        print("\nAnnotation Schema:")
        for ann in data["annotations"]:
            print(f"  Metric: {ann.get('metric')}")
            print(f"  Category: {ann.get('category')}")
            print(f"  Labels: {ann.get('labels_list')}")
            print(f"  Prompt: {ann.get('prompt')[:100]}..." if ann.get('prompt') else "  Prompt: N/A")
    
    # Record structure
    if "records" in data and data["records"]:
        print("\nRecord structure (first record):")
        record = data["records"][0]
        for key, value in record.items():
            if key == "prompt":
                print(f"  {key}: (prompt text, {len(str(value))} chars)")
            elif key == "tags":
                print(f"  {key}: {value}")
            elif isinstance(value, list) and len(value) > 0:
                print(f"  {key}: (list with {len(value)} items)")
                if isinstance(value[0], dict):
                    print(f"      Example: {list(value[0].keys())}")
            elif isinstance(value, dict):
                print(f"  {key}: (dict with keys: {list(value.keys())[:5]})")
            else:
                print(f"  {key}: {value}")


def explore_labels(data):
    """Analyze label distribution."""
    print("\n" + "=" * 70)
    print("LABEL ANALYSIS")
    print("=" * 70)
    
    if "records" not in data:
        return
    
    records = data["records"]
    label_counts = Counter()
    tag_counts = defaultdict(int)
    
    for record in records:
        if "human_label" in record:
            label_counts[record["human_label"]] += 1
        if "tags" in record and record["tags"]:
            for tag in record["tags"]:
                tag_counts[tag] += 1
    
    print(f"\nTotal samples: {len(records)}")
    print(f"\nLabel distribution (human_label):")
    for label, count in sorted(label_counts.items()):
        percentage = (count / len(records)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    if tag_counts:
        print(f"\nTag distribution (dataset split):")
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
            percentage = (count / len(records)) * 100
            print(f"  {tag}: {count} ({percentage:.1f}%)")


def explore_sample_content(data):
    """Display sample records with context."""
    print("\n" + "=" * 70)
    print("SAMPLE RECORDS")
    print("=" * 70)
    
    if "records" not in data or not data["records"]:
        return
    
    records = data["records"]
    
    # Show 3 samples: positive, negative, mixed
    label_groups = defaultdict(list)
    for i, record in enumerate(records):
        label = record.get("human_label", "unknown")
        label_groups[label].append(i)
    
    for label in sorted(label_groups.keys())[:2]:  # Show 2 different labels
        indices = label_groups[label]
        if indices:
            idx = indices[0]
            record = records[idx]
            
            print(f"\nSample #{idx + 1} - Label: {label}")
            print("-" * 70)
            
            # Show prompt (truncated)
            if "prompt" in record:
                prompt = record["prompt"]
                print(f"Prompt (first 300 chars):\n{prompt[:300]}...")
            
            # Show metadata
            if "tags" in record:
                print(f"Tags: {record['tags']}")
            if "sample_id" in record:
                print(f"Sample ID: {record['sample_id']}")


def main():
    print("\n" + "=" * 70)
    print("QAGS DATASET EXPLORATION")
    print("=" * 70)
    
    data = load_qags_data()
    if data is None:
        return
    
    explore_structure(data)
    explore_labels(data)
    explore_sample_content(data)
    
    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
