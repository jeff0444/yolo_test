import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
from collections import defaultdict
import datetime
from pathlib import Path

class DetectionStats:
    def __init__(self, source_name, model_name, input_size=(640, 640), conf_threshold=None, class_names=None):
        self.source_name = Path(source_name).name
        self.model_name = Path(model_name).name
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.class_names = class_names if class_names else {} # dict {id: name}

        # data[class_name] = list of dicts {'conf': float, 'min_dim': float, 'orig_w': int, 'orig_h': int}
        self.data = defaultdict(list)

        self.bins = [0, 5, 8, 12, 16, 24, 32, 48, 64, 128, float('inf')]
        self.bin_labels = ['0-5', '5-8', '8-12', '12-16', '16-24', '24-32', '32-48', '48-64', '64-128', '>128']

    def update(self, results, img_shape):
        """
        results: ultralytics Results object
        img_shape: (h, w) of the original image
        """
        h_orig, w_orig = img_shape[:2]
        # Calculate scale to model input
        scale = min(self.input_size[0] / h_orig, self.input_size[1] / w_orig)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                conf = float(box.conf[0])

                # Box in xywh format
                w = float(box.xywh[0][2])
                h = float(box.xywh[0][3])

                min_dim_model = min(w * scale, h * scale)

                self.data[class_name].append({
                    'conf': conf,
                    'min_dim': min_dim_model,
                    'orig_w': w,
                    'orig_h': h
                })

    def save_report(self, output_path):
        print(f"Generating PDF report: {output_path}")
        try:
            with PdfPages(output_path) as pdf:
                # Page 1: Summary (Portrait A4)
                plt.figure(figsize=(8.27, 11.69)) # A4 Portrait
                plt.axis('off')

                text = "Detection Report\n\n"
                text += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                text += f"Source: {self.source_name}\n"
                text += f"Model: {self.model_name}\n"
                text += f"Model Input Size: {self.input_size}\n"
                text += f"Confidence Threshold: {self.conf_threshold}\n\n"

                text += "Class Statistics:\n"

                sorted_ids = sorted(self.class_names.keys())

                classes_text = []
                for cls_id in sorted_ids:
                    name = self.class_names[cls_id]
                    count = len(self.data[name])
                    classes_text.append(f"{cls_id}: {name} ({count})")

                summary_content = text + "\n".join(classes_text)

                plt.text(0.01, 0.99, summary_content, fontsize=12, verticalalignment='top', fontfamily='monospace',
                         wrap=True)

                pdf.savefig()
                plt.close()

                # Subsequent Pages: Per Class Stats (Multi-plot)
                # Filter active classes and sort by Model Index
                active_classes_names = set(name for name, items in self.data.items() if len(items) > 0)
                sorted_ids = sorted(self.class_names.keys())
                classes_to_plot = []
                for cls_id in sorted_ids:
                    name = self.class_names[cls_id]
                    if name in active_classes_names:
                        classes_to_plot.append(name)

                # 6 plots per page (3 rows x 2 cols) - Portrait Layout
                plots_per_page = 6
                rows = 3
                cols = 2

                for i in range(0, len(classes_to_plot), plots_per_page):
                    batch_classes = classes_to_plot[i : i + plots_per_page]

                    fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69)) # Portrait A4
                    axes_flat = axes.flatten()

                    # Hide unused subplots
                    for j in range(len(batch_classes), len(axes_flat)):
                        axes_flat[j].axis('off')

                    for ax_idx, cls_name in enumerate(batch_classes):
                        ax1 = axes_flat[ax_idx]
                        items = self.data[cls_name]
                        min_dims = np.array([item['min_dim'] for item in items])
                        confs = np.array([item['conf'] for item in items])

                        bin_indices = np.digitize(min_dims, self.bins) - 1

                        boxplot_data = []
                        counts = []

                        for bin_idx in range(len(self.bin_labels)):
                            mask = (bin_indices == bin_idx)
                            bin_confs = confs[mask]
                            boxplot_data.append(bin_confs)
                            counts.append(len(bin_confs))

                        # Plot 1: Box Plot (Confidence) - Left Axis
                        bp = ax1.boxplot(boxplot_data, positions=range(len(self.bin_labels)), patch_artist=True,
                                         showfliers=False, widths=0.6)

                        for patch in bp['boxes']:
                            patch.set_facecolor('lightblue')
                            patch.set_alpha(0.5)
                        for median in bp['medians']:
                            median.set_color('blue')

                        ax1.set_ylabel('Conf', color='blue', fontsize=8)
                        ax1.tick_params(axis='y', labelcolor='blue', labelsize=7)
                        ax1.set_ylim(0, 1.05)
                        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

                        # Annotate Median, Q1, Q3
                        for bin_idx, bin_confs in enumerate(boxplot_data):
                            if len(bin_confs) > 0:
                                q1 = np.percentile(bin_confs, 25)
                                med = np.median(bin_confs)
                                q3 = np.percentile(bin_confs, 75)

                                font_size = 5
                                ax1.text(bin_idx, q3 + 0.02, f'{q3:.2f}', ha='center', va='bottom', fontsize=font_size, color='navy')
                                ax1.text(bin_idx, med, f'{med:.2f}', ha='center', va='center', fontsize=font_size, color='red', fontweight='bold')

                                # ax1.text(bin_idx, q1 - 0.02, f'{q1:.2f}', ha='center', va='top', fontsize=font_size, color='navy')

                        # Plot 2: Line Plot (Count) - Right Axis
                        ax2 = ax1.twinx()
                        ax2.plot(range(len(self.bin_labels)), counts, color='red', marker='o', linestyle='-', linewidth=1.5, markersize=3)
                        ax2.set_ylabel('Count', color='red', fontsize=8)
                        ax2.tick_params(axis='y', labelcolor='red', labelsize=7)

                        # Adjust Right Y-Axis limit to look nice
                        max_count = max(counts) if counts else 0
                        ax2.set_ylim(0, max_count * 1.5 if max_count > 0 else 10)

                        # Annotate Counts
                        for bin_idx, count in enumerate(counts):
                            if count > 0:
                                ax2.text(bin_idx, count, str(count), ha='center', va='bottom', fontsize=6, fontweight='bold', color='red',
                                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

                        ax1.set_xticks(range(len(self.bin_labels)))
                        ax1.set_xticklabels(self.bin_labels, rotation=45, fontsize=7)
                        ax1.set_title(f'{cls_name} (n={len(items)})', fontsize=10)
                        # ax1.set_xlabel('Min Dim (px)', fontsize=8)

                    plt.tight_layout()
                    # Add gap between subplots
                    plt.subplots_adjust(hspace=0.4, wspace=0.3)
                    pdf.savefig(fig)
                    plt.close(fig)

            print(f"Report saved successfully.")
        except Exception as e:
            print(f"Failed to save PDF report: {e}")