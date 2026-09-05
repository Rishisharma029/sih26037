import os
import json
from collections import Counter
from typing import Dict, Any

class IDDDatasetAnalyzer:
    def __init__(self, base_data_dir: str):
        self.base_data_dir = base_data_dir

    def analyze(self) -> Dict[str, Any]:
        splits = ["train", "val", "test"]
        summary = {
            "total_images": 0,
            "splits": {},
            "class_distribution": Counter(),
            "resolutions": Counter(),
            "total_bounding_boxes": 0
        }

        for split in splits:
            manifest_path = os.path.join(self.base_data_dir, f"{split}_manifest.json")
            if not os.path.exists(manifest_path):
                continue

            with open(manifest_path, "r", encoding="utf-8") as f:
                samples = json.load(f)

            split_box_count = 0
            split_class_dist = Counter()

            for s in samples:
                summary["resolutions"][f"{s['width']}x{s['height']}"] += 1
                for b in s.get("boxes", []):
                    cls_lbl = b["label"]
                    summary["class_distribution"][cls_lbl] += 1
                    split_class_dist[cls_lbl] += 1
                    split_box_count += 1

            summary["total_images"] += len(samples)
            summary["total_bounding_boxes"] += split_box_count
            summary["splits"][split] = {
                "num_images": len(samples),
                "num_boxes": split_box_count,
                "class_counts": dict(split_class_dist)
            }

        summary["class_distribution"] = dict(summary["class_distribution"])
        summary["resolutions"] = dict(summary["resolutions"])
        return summary

    def generate_report_markdown(self, output_path: str = None) -> str:
        stats = self.analyze()
        md = f"""# IDD (India Driving Dataset) Analysis Report
## Perception & Road User Distribution

- **Total Images**: {stats['total_images']} (Train: {stats['splits'].get('train', {}).get('num_images', 0)}, Val: {stats['splits'].get('val', {}).get('num_images', 0)}, Test: {stats['splits'].get('test', {}).get('num_images', 0)})
- **Total Labeled Objects**: {stats['total_bounding_boxes']}
- **Image Resolutions**: {list(stats['resolutions'].keys())}

### Class Distribution (8 Indian Road Actor Classes)
| Class | Total Instances | Percentage |
|---|---|---|
"""
        tot = max(1, stats['total_bounding_boxes'])
        for cls_name, count in sorted(stats['class_distribution'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / tot) * 100.0
            md += f"| `{cls_name}` | {count} | {pct:.1f}% |\n"

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"Report written to {output_path}")

        return md

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    analyzer = IDDDatasetAnalyzer(base_dir)
    print(analyzer.generate_report_markdown())
