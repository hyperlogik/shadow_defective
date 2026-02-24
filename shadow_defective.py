import sys
import base64
import mimetypes
import io
import datetime
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QTextEdit,
                             QFileDialog, QMessageBox, QGroupBox, QProgressBar)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from openai import OpenAI
from PIL import Image

DEFAULT_PROMPT = (
    "Based on the length and direction of the shadows, "
    "the architectural landmarks, and the known latitude/longitude "
    "({lat}, {lon}), calculate the approximate solar angle "
    "and time of day. Walk me through your reasoning."
)
MAX_IMAGE_DIMENSION = 1920


class APIWorker(QThread):
    """A separate thread to handle the OpenAI API request so the main GUI doesn't freeze."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, api_key, image_path, lat, lon, prompt_template):
        super().__init__()
        self.api_key = api_key
        self.image_path = image_path
        self.lat = lat
        self.lon = lon
        self.prompt_template = prompt_template

    def encode_image(self, image_path):
        """Open, resize if necessary, and base64-encode the image."""
        img = Image.open(image_path)
        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))

        # Preserve format for MIME detection; fall back to JPEG
        fmt = img.format or "JPEG"
        buffer = io.BytesIO()
        # PIL may lose the format attribute after thumbnail(); re-use original ext
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
            fmt = "JPEG"

        # Convert mode if needed (e.g. RGBA → RGB for JPEG)
        if mime_type == "image/jpeg" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        img.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), mime_type

    def run(self):
        try:
            # Only read the key at call time
            api_key = self.api_key
            client = OpenAI(api_key=api_key)

            base64_image, mime_type = self.encode_image(self.image_path)

            prompt_text = self.prompt_template.replace("{lat}", self.lat).replace("{lon}", self.lon)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            result_text = response.choices[0].message.content
            self.finished.emit(result_text)

        except Exception as e:
            self.error.emit(str(e))


class ShadowDetectiveApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.worker = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Shadow Defective")
        self.setMinimumSize(660, 850)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)

        # --- API Key Section ---
        api_layout = QHBoxLayout()
        self.api_input = QLineEdit()
        self.api_input.setPlaceholderText("Enter your OpenAI API Key (sk-...)")
        self.api_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_layout.addWidget(QLabel("API Key:"))
        api_layout.addWidget(self.api_input)
        main_layout.addLayout(api_layout)

        # --- Image Selection Section ---
        img_group = QGroupBox("Target Image")
        img_layout = QVBoxLayout()

        self.image_label = QLabel("No image selected.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px dashed #aaa; padding: 10px;")
        self.image_label.setMinimumHeight(250)

        self.btn_select_img = QPushButton("Select Photograph")
        self.btn_select_img.clicked.connect(self.select_image)

        img_layout.addWidget(self.image_label)
        img_layout.addWidget(self.btn_select_img)
        img_group.setLayout(img_layout)
        main_layout.addWidget(img_group)

        # --- Location Data Section ---
        loc_group = QGroupBox("Metadata / Coordinates")
        loc_layout = QHBoxLayout()

        self.lat_input = QLineEdit()
        self.lat_input.setPlaceholderText("e.g., 48.8584")
        self.lon_input = QLineEdit()
        self.lon_input.setPlaceholderText("e.g., 2.2945")

        loc_layout.addWidget(QLabel("Latitude:"))
        loc_layout.addWidget(self.lat_input)
        loc_layout.addWidget(QLabel("Longitude:"))
        loc_layout.addWidget(self.lon_input)
        loc_group.setLayout(loc_layout)
        main_layout.addWidget(loc_group)

        # --- Prompt Section ---
        prompt_group = QGroupBox("Analysis Prompt  (use {lat} and {lon} as placeholders)")
        prompt_layout = QVBoxLayout()
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText(DEFAULT_PROMPT)
        self.prompt_edit.setFixedHeight(90)
        btn_reset_prompt = QPushButton("Reset to Default")
        btn_reset_prompt.clicked.connect(lambda: self.prompt_edit.setPlainText(DEFAULT_PROMPT))
        prompt_layout.addWidget(self.prompt_edit)
        prompt_layout.addWidget(btn_reset_prompt)
        prompt_group.setLayout(prompt_layout)
        main_layout.addWidget(prompt_group)

        # --- Analysis / Progress ---
        self.btn_analyze = QPushButton("Run Shadow Analysis")
        self.btn_analyze.setStyleSheet(
            "background-color: #2E86C1; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_analyze.clicked.connect(self.run_analysis)
        main_layout.addWidget(self.btn_analyze)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)   # indeterminate spinner
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # --- Results Section ---
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setPlaceholderText("Analysis results will appear here...")

        self.btn_save_report = QPushButton("Save Report…")
        self.btn_save_report.setEnabled(False)
        self.btn_save_report.clicked.connect(self.save_report)

        results_layout.addWidget(self.result_box)
        results_layout.addWidget(self.btn_save_report)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        self.setLayout(main_layout)

    # ------------------------------------------------------------------
    # Image selection
    # ------------------------------------------------------------------

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if file_name:
            self.image_path = file_name
            self._refresh_image_preview()

            # Attempt to auto-populate GPS coords from EXIF
            self._try_populate_gps(file_name)

    def _refresh_image_preview(self):
        if not self.image_path:
            return
        pixmap = QPixmap(self.image_path)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_image_preview()

    def _try_populate_gps(self, image_path):
        """Auto-populate lat/lon from image EXIF if available."""
        try:
            from PIL.ExifTags import TAGS, GPSTAGS
            img = Image.open(image_path)
            exif_data = img._getexif()
            if not exif_data:
                return

            gps_info = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    for gps_tag_id, gps_val in value.items():
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_info[gps_tag] = gps_val

            if not gps_info:
                return

            def dms_to_dd(dms, ref):
                degrees, minutes, seconds = dms
                dd = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
                if ref in ("S", "W"):
                    dd = -dd
                return round(dd, 6)

            lat = dms_to_dd(gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N"))
            lon = dms_to_dd(gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E"))
            self.lat_input.setText(str(lat))
            self.lon_input.setText(str(lon))

        except Exception:
            pass  # EXIF extraction is best-effort

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def run_analysis(self):
        # Read the key only at call time — don't cache it on self
        api_key = self.api_input.text().strip()
        lat = self.lat_input.text().strip()
        lon = self.lon_input.text().strip()
        prompt = self.prompt_edit.toPlainText().strip()

        if not api_key:
            QMessageBox.warning(self, "Error", "Please enter your OpenAI API key.")
            return
        if not self.image_path:
            QMessageBox.warning(self, "Error", "Please select an image first.")
            return
        if not lat or not lon:
            QMessageBox.warning(self, "Error", "Please provide both latitude and longitude.")
            return
        if not prompt:
            QMessageBox.warning(self, "Error", "Prompt cannot be empty.")
            return

        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setText("Analyzing shadows…")
        self.progress_bar.setVisible(True)
        self.btn_save_report.setEnabled(False)
        self.result_box.clear()

        self.worker = APIWorker(api_key, self.image_path, lat, lon, prompt)
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.worker.start()

    def on_analysis_complete(self, result):
        self.result_box.setText(result)
        self.btn_save_report.setEnabled(True)
        self._reset_button()

    def on_analysis_error(self, error_msg):
        # Give a more specific hint based on the error string
        if "401" in error_msg or "Incorrect API key" in error_msg or "invalid_api_key" in error_msg:
            user_msg = "Authentication failed. Please check your OpenAI API key."
        elif "429" in error_msg or "Rate limit" in error_msg:
            user_msg = "Rate limit or quota exceeded. Wait a moment or check your OpenAI billing."
        elif "Connection" in error_msg or "Network" in error_msg or "timeout" in error_msg.lower():
            user_msg = "Network error. Check your internet connection and try again."
        else:
            user_msg = error_msg

        QMessageBox.critical(self, "API Error", f"An error occurred:\n{user_msg}")
        self._reset_button()

    def _reset_button(self):
        self.btn_analyze.setEnabled(True)
        self.btn_analyze.setText("Run Shadow Analysis")
        self.progress_bar.setVisible(False)

    # ------------------------------------------------------------------
    # Report export
    # ------------------------------------------------------------------

    def save_report(self):
        result_text = self.result_box.toPlainText().strip()
        if not result_text:
            QMessageBox.warning(self, "Nothing to Save", "Run an analysis first.")
            return

        default_name = f"shadow_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Report", default_name, "Text files (*.txt);;Markdown files (*.md)"
        )
        if not file_name:
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = (
            f"Shadow Defective OSINT Report\n"
            f"{'=' * 40}\n"
            f"Timestamp  : {timestamp}\n"
            f"Image      : {self.image_path or 'N/A'}\n"
            f"Latitude   : {self.lat_input.text().strip()}\n"
            f"Longitude  : {self.lon_input.text().strip()}\n"
            f"{'=' * 40}\n\n"
            f"PROMPT\n"
            f"{'-' * 40}\n"
            f"{self.prompt_edit.toPlainText().strip()}\n\n"
            f"ANALYSIS\n"
            f"{'-' * 40}\n"
            f"{result_text}\n"
        )

        try:
            Path(file_name).write_text(report, encoding="utf-8")
            QMessageBox.information(self, "Saved", f"Report saved to:\n{file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save report:\n{e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ShadowDetectiveApp()
    ex.show()
    sys.exit(app.exec())


