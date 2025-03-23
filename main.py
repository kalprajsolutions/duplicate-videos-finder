import sys
import os
import shutil
import cv2
import numpy as np
import imagehash
from PIL import Image
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import subprocess

from sklearn.cluster import DBSCAN

from PyQt5 import QtWidgets, QtGui, QtCore

# -------------------------------
# Video Processing Functions
# -------------------------------
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm")


def extract_keyframes(video_path, num_frames=3):
    """Extracts keyframes from a video. Returns list of frames in BGR."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Failed to open {video_path}")
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        print(f"⚠️ No valid frames in {video_path}")
        cap.release()
        return []
    frames = []
    # Use positions at 10%, 50%, 90% of video
    positions = [int(frame_count * p) for p in [0.1, 0.5, 0.9] if frame_count * p > 0]
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            # For feature extraction, we use a smaller resized version.
            frame_small = cv2.resize(frame, (64, 64))
            frames.append(frame_small)
    cap.release()
    if not frames:
        print(f"⚠️ No frames extracted from {video_path}")
    return frames if frames else [np.zeros((64, 64, 3), dtype=np.uint8)]


def compute_video_features(video_path):
    """Computes perceptual hash (pHash) for each keyframe (converted to grayscale)."""
    keyframes = extract_keyframes(video_path)
    if not keyframes:
        return None
    phashes = [imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) for frame in keyframes]
    return phashes


def compare_videos(video1_features, video2_features):
    """Computes average difference between video keyframe pHashes."""
    if not video1_features or not video2_features:
        return 1  # maximal difference if feature extraction fails
    min_length = min(len(video1_features), len(video2_features))
    phash_diff = sum(video1_features[i] - video2_features[i] for i in range(min_length))
    return phash_diff / min_length


def find_video_files(directory):
    """Scans directory and returns all video file paths."""
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                full_path = os.path.abspath(os.path.join(root, file))
                video_files.append(full_path)
    return video_files


def cluster_videos(video_files, eps_threshold=8):
    """Clusters videos based on pHash differences using DBSCAN."""
    with Pool(cpu_count()) as pool:
        video_features = pool.map(compute_video_features, video_files)
    valid_videos = []
    feature_dict = {}
    for i, features in enumerate(video_features):
        if features is not None and len(features) > 0:
            valid_videos.append(video_files[i])
            feature_dict[video_files[i]] = features
        else:
            print(f"⚠️ Skipping {video_files[i]} due to feature extraction failure")
    if not valid_videos:
        print("❌ No valid videos for clustering")
        return {}
    num_videos = len(valid_videos)
    similarity_matrix = np.zeros((num_videos, num_videos))
    for i in range(num_videos):
        for j in range(i + 1, num_videos):
            similarity = compare_videos(feature_dict[valid_videos[i]], feature_dict[valid_videos[j]])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    clustering = DBSCAN(eps=eps_threshold, min_samples=2, metric="precomputed").fit(similarity_matrix)
    clusters = defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        clusters[label].append(valid_videos[i])
    return clusters


# -------------------------------
# QThread Worker for Clustering
# -------------------------------
class ClusterWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(dict)  # emits clusters dictionary

    def __init__(self, video_dir, parent=None):
        super().__init__(parent)
        self.video_dir = video_dir

    def run(self):
        video_files = find_video_files(self.video_dir)
        if not video_files:
            self.finished.emit({})
            return
        clusters = cluster_videos(video_files)
        self.finished.emit(clusters)


# -------------------------------
# Thumbnail and File Info Helpers
# -------------------------------
def get_color_thumbnail(video_path):
    """Extract first keyframe in color and return a QPixmap."""
    frames = extract_keyframes(video_path)
    if frames:
        frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        height, width, ch = frame.shape
        bytes_per_line = ch * width
        q_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_image).scaled(100, 100, QtCore.Qt.KeepAspectRatio,
                                                       QtCore.Qt.SmoothTransformation)
    return None


def get_file_size_str(video_path):
    """Returns file size in MB (human readable)."""
    try:
        size_bytes = os.path.getsize(video_path)
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    except Exception as e:
        return "N/A"


# -------------------------------
# UI Components: Cluster Table Widget
# -------------------------------
class ClusterTableWidget(QtWidgets.QGroupBox):
    def __init__(self, title, video_list, parent=None):
        super().__init__(title, parent)
        self.video_list = video_list
        self.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4CAF50;
                border-radius: 8px;
                margin-top: 10px;
                font: bold 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)
        self.init_ui()

    def init_ui(self):
        vlayout = QtWidgets.QVBoxLayout(self)
        # Header with "Select/Deselect All" button
        header_layout = QtWidgets.QHBoxLayout()
        self.select_all_btn = QtWidgets.QPushButton("Select/Deselect All")
        self.select_all_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 4px 8px; border-radius: 4px;")
        self.select_all_btn.clicked.connect(self.toggle_select_all)
        header_layout.addWidget(self.select_all_btn)
        header_layout.addStretch()
        vlayout.addLayout(header_layout)

        # Create table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Thumbnail", "File Name", "Path", "Size", "Open", "Select"])
        self.table.horizontalHeader().setStyleSheet("font-weight: bold;")
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setRowCount(len(self.video_list))
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget { background-color: #f9f9f9; alternate-background-color: #e9e9e9; }
            QTableWidget::item { padding: 4px; }
        """)

        for row, video in enumerate(self.video_list):
            # Thumbnail
            thumb = get_color_thumbnail(video)
            thumb_label = QtWidgets.QLabel()
            thumb_label.setAlignment(QtCore.Qt.AlignCenter)
            if thumb:
                thumb_label.setPixmap(thumb)
            else:
                thumb_label.setText("No Image")
            self.table.setCellWidget(row, 0, thumb_label)

            # File Name
            filename_item = QtWidgets.QTableWidgetItem(os.path.basename(video))
            self.table.setItem(row, 1, filename_item)

            # Path
            path_item = QtWidgets.QTableWidgetItem(video)
            self.table.setItem(row, 2, path_item)

            # Size
            size_item = QtWidgets.QTableWidgetItem(get_file_size_str(video))
            self.table.setItem(row, 3, size_item)

            # Open Button – opens the exact file in explorer.
            open_btn = QtWidgets.QPushButton("Open")
            open_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 4px 6px; border-radius: 4px;")
            open_btn.clicked.connect(lambda checked, v=video: self.open_in_explorer(v))
            self.table.setCellWidget(row, 4, open_btn)

            # Checkbox for selection
            checkbox = QtWidgets.QCheckBox()
            checkbox.setStyleSheet("margin-left:40%; margin-right:40%;")
            self.table.setCellWidget(row, 5, checkbox)
            self.table.setRowHeight(row, 110)

        # Resize table height to show all rows
        total_height = self.table.horizontalHeader().height()
        for row in range(self.table.rowCount()):
            total_height += self.table.rowHeight(row)
        self.table.setFixedHeight(total_height + 2)

        vlayout.addWidget(self.table)
        self.setLayout(vlayout)

    def toggle_select_all(self):
        total = self.table.rowCount()
        select_all = any(
            not isinstance(self.table.cellWidget(row, 5), QtWidgets.QCheckBox) or not self.table.cellWidget(row,
                                                                                                            5).isChecked()
            for row in range(total)
        )
        for row in range(total):
            widget = self.table.cellWidget(row, 5)
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(select_all)

    def open_in_explorer(self, video):
        try:
            if sys.platform == "win32":
                subprocess.Popen(f'explorer /select,"{video}"')
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", video])
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(video)])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not open file explorer: {e}")

    def get_selected_videos(self):
        selected = []
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 5)
            if isinstance(widget, QtWidgets.QCheckBox) and widget.isChecked():
                video = self.table.item(row, 2).text()
                selected.append(video)
        return selected

    def auto_select_by_size(self, keep_smallest=True):
        rows = []
        for row in range(self.table.rowCount()):
            video = self.table.item(row, 2).text()
            try:
                size = os.path.getsize(video)
            except Exception:
                size = float('inf')
            rows.append((row, video, size))
        if not rows:
            return
        rows_sorted = sorted(rows, key=lambda x: x[2], reverse=not keep_smallest)
        keep_row = rows_sorted[0][0]
        for row, video, size in rows:
            checkbox = self.table.cellWidget(row, 5)
            if row != keep_row:
                checkbox.setChecked(True)
            else:
                checkbox.setChecked(False)


# -------------------------------
# Main Window UI
# -------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, video_dir):
        super().__init__()
        self.video_dir = video_dir
        self.cluster_widgets = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Duplicate Video Remover")
        self.setStyleSheet("""
            QMainWindow { background-color: #ffffff; }
            QPushButton { font: 12px; }
        """)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # Top button panel
        button_layout = QtWidgets.QHBoxLayout()
        self.scan_button = QtWidgets.QPushButton("Scan & Cluster Videos")
        self.scan_button.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 6px 12px; border-radius: 6px;")
        self.scan_button.clicked.connect(self.scan_videos)
        button_layout.addWidget(self.scan_button)

        self.delete_button = QtWidgets.QPushButton("Delete Selected Videos")
        self.delete_button.setStyleSheet(
            "background-color: #f44336; color: white; padding: 6px 12px; border-radius: 6px;")
        self.delete_button.clicked.connect(self.delete_selected)
        self.delete_button.setEnabled(False)
        button_layout.addWidget(self.delete_button)

        # Auto-select options
        self.radio_keep_smallest = QtWidgets.QRadioButton("Keep Smallest (Select larger duplicates)")
        self.radio_keep_smallest.setChecked(True)
        self.radio_keep_largest = QtWidgets.QRadioButton("Keep Largest (Select smaller duplicates)")
        button_layout.addWidget(self.radio_keep_smallest)
        button_layout.addWidget(self.radio_keep_largest)

        self.auto_select_button = QtWidgets.QPushButton("Auto Select Duplicates by Size")
        self.auto_select_button.setStyleSheet(
            "background-color: #FF9800; color: white; padding: 6px 12px; border-radius: 6px;")
        self.auto_select_button.clicked.connect(self.auto_select_duplicates)
        button_layout.addWidget(self.auto_select_button)

        main_layout.addLayout(button_layout)

        # Scroll Area for clusters
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setStyleSheet("background-color: #ffffff;")
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)

        self.status_bar = self.statusBar()
        self.resize(1200, 800)

    def scan_videos(self):
        self.scan_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.clear_clusters()
        self.status_bar.showMessage("Scanning videos and clustering, please wait...")

        self.progress_dialog = QtWidgets.QProgressDialog("Processing videos...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Please Wait")
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dialog.show()

        self.worker = ClusterWorker(self.video_dir)
        self.worker.finished.connect(self.on_clustering_finished)
        self.worker.start()

    def clear_clusters(self):
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.cluster_widgets = []

    def on_clustering_finished(self, clusters):
        self.progress_dialog.close()
        self.scan_button.setEnabled(True)
        duplicate_clusters = {k: v for k, v in clusters.items() if k != -1 and len(v) > 1}
        unique_videos = clusters.get(-1, [])

        if duplicate_clusters:
            for cluster_id, videos in duplicate_clusters.items():
                title = f"Duplicate Group {cluster_id} ({len(videos)} videos)"
                table_widget = ClusterTableWidget(title, videos)
                self.scroll_layout.addWidget(table_widget)
                self.cluster_widgets.append(table_widget)
        else:
            label = QtWidgets.QLabel("No duplicate video groups found.")
            self.scroll_layout.addWidget(label)

        # if unique_videos:
        #     title = f"Unique Videos ({len(unique_videos)} videos)"
        #     table_widget = ClusterTableWidget(title, unique_videos)
        #     self.scroll_layout.addWidget(table_widget)
        #     self.cluster_widgets.append(table_widget)

        self.delete_button.setEnabled(True)
        self.status_bar.showMessage("Clustering complete.", 5000)

    def auto_select_duplicates(self):
        keep_smallest = self.radio_keep_smallest.isChecked()
        for widget in self.cluster_widgets:
            if widget.table.rowCount() > 1:
                widget.auto_select_by_size(keep_smallest=keep_smallest)

    def delete_selected(self):
        to_delete = []
        for widget in self.cluster_widgets:
            to_delete.extend(widget.get_selected_videos())
        if not to_delete:
            QtWidgets.QMessageBox.information(self, "No Selection", "No videos selected for deletion.")
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to permanently delete {len(to_delete)} selected video(s)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            for video in to_delete:
                try:
                    os.remove(video)
                except Exception as e:
                    print(f"Error deleting {video}: {e}")
            QtWidgets.QMessageBox.information(self, "Deletion Complete", "Selected videos have been deleted.")
            self.scan_videos()


if __name__ == "__main__":
    VIDEO_DIR = os.path.abspath("videos")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(VIDEO_DIR)
    window.show()
    sys.exit(app.exec_())
