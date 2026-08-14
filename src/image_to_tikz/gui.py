from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .golden_prompt import build_golden_prompt
from .pipeline import analyze_image
from .tikz_verifier import compile_and_compare


APP_NAME = "Image-to-TikZ Studio"
CONFIG_PATH = Path(os.environ.get("APPDATA", Path.home())) / "ImageToTikZ" / "settings.json"


def run() -> int:
    try:
        from PySide6.QtCore import QProcess, QSettings, Qt
        from PySide6.QtWidgets import (
            QApplication,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QPlainTextEdit,
            QSpinBox,
            QDoubleSpinBox,
            QTabWidget,
            QVBoxLayout,
            QWidget,
            QCheckBox,
        )
    except ImportError as exc:
        raise SystemExit("GUI requires optional dependency: pip install -e '.[gui]'") from exc

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(APP_NAME)
            self.resize(1180, 820)
            self.server = QProcess(self)
            self.settings = QSettings("ImageToTikZ", "Studio")
            self._build()
            self._load()

        def _build(self) -> None:
            root = QWidget()
            self.setCentralWidget(root)
            outer = QVBoxLayout(root)

            files = QGroupBox("Input / Models")
            grid = QGridLayout(files)
            self.image = QLineEdit()
            self.server_exe = QLineEdit()
            self.model = QLineEdit()
            self.mmproj = QLineEdit()
            self.output_dir = QLineEdit()
            self.pdflatex = QLineEdit("pdflatex")
            self.pdftoppm = QLineEdit("pdftoppm")
            self._browse_row(grid, 0, "Image", self.image, self._pick_image)
            self._browse_row(grid, 1, "llama-server.exe", self.server_exe, self._pick_exe)
            self._browse_row(grid, 2, "SmolVLM2 GGUF", self.model, self._pick_file)
            self._browse_row(grid, 3, "mmproj GGUF", self.mmproj, self._pick_file)
            self._browse_row(grid, 4, "Output folder", self.output_dir, self._pick_dir)
            self._browse_row(grid, 5, "pdflatex", self.pdflatex, self._pick_exe)
            self._browse_row(grid, 6, "pdftoppm", self.pdftoppm, self._pick_exe)
            outer.addWidget(files)

            settings = QGroupBox("Pipeline / llama.cpp")
            form = QFormLayout(settings)
            self.port = QSpinBox(); self.port.setRange(1, 65535); self.port.setValue(8080)
            self.ctx = QSpinBox(); self.ctx.setRange(512, 32768); self.ctx.setValue(4096)
            self.ngl = QSpinBox(); self.ngl.setRange(0, 200); self.ngl.setValue(99)
            self.crops = QSpinBox(); self.crops.setRange(0, 32); self.crops.setValue(8)
            self.temp = QDoubleSpinBox(); self.temp.setRange(0.0, 2.0); self.temp.setSingleStep(0.05); self.temp.setValue(0.1)
            self.top_p = QDoubleSpinBox(); self.top_p.setRange(0.01, 1.0); self.top_p.setSingleStep(0.01); self.top_p.setValue(0.9)
            self.ocr = QCheckBox("Enable OCR")
            self.multiscale = QCheckBox("Multiscale analysis"); self.multiscale.setChecked(True)
            form.addRow("Port", self.port); form.addRow("Context", self.ctx); form.addRow("GPU layers", self.ngl)
            form.addRow("Semantic crops", self.crops); form.addRow("Smol temperature", self.temp); form.addRow("Smol top-p", self.top_p)
            form.addRow(self.ocr); form.addRow(self.multiscale)
            outer.addWidget(settings)

            llm = QGroupBox("Downstream commercial LLM (optional)")
            lform = QFormLayout(llm)
            self.llm_endpoint = QLineEdit("https://api.openai.com/v1")
            self.llm_model = QLineEdit()
            self.llm_key = QLineEdit(); self.llm_key.setEchoMode(QLineEdit.Password)
            self.llm_endpoint.setPlaceholderText("OpenAI-compatible endpoint; leave empty to export prompt only")
            lform.addRow("Endpoint", self.llm_endpoint); lform.addRow("Model", self.llm_model); lform.addRow("API key", self.llm_key)
            outer.addWidget(llm)

            buttons = QHBoxLayout()
            for text, slot in [
                ("Start llama.cpp", self.start_server), ("Stop llama.cpp", self.stop_server),
                ("Analyze image", self.analyze), ("Copy Golden Prompt", self.copy_prompt),
                ("Verify TikZ", self.verify), ("Save settings", self.save),
            ]:
                b = QPushButton(text); b.clicked.connect(slot); buttons.addWidget(b)
            outer.addLayout(buttons)

            self.tabs = QTabWidget()
            self.context_box = QPlainTextEdit(); self.context_box.setReadOnly(True)
            self.prompt_box = QPlainTextEdit(); self.prompt_box.setReadOnly(True)
            self.tikz_box = QPlainTextEdit()
            self.log_box = QPlainTextEdit(); self.log_box.setReadOnly(True)
            self.tabs.addTab(self.context_box, "Visual Context")
            self.tabs.addTab(self.prompt_box, "Golden Prompt")
            self.tabs.addTab(self.tikz_box, "TikZ")
            self.tabs.addTab(self.log_box, "Log")
            outer.addWidget(self.tabs, 1)
            self.server.readyReadStandardOutput.connect(lambda: self.log_box.appendPlainText(bytes(self.server.readAllStandardOutput()).decode(errors="replace")))
            self.server.readyReadStandardError.connect(lambda: self.log_box.appendPlainText(bytes(self.server.readAllStandardError()).decode(errors="replace")))

        def _browse_row(self, grid, row, label, edit, callback):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            b = QPushButton("Browse")
            b.clicked.connect(callback)
            grid.addWidget(b, row, 2)

        def _pick_image(self):
            p, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
            if p: self.image.setText(p)

        def _pick_file(self):
            p, _ = QFileDialog.getOpenFileName(self, "Select file", "", "GGUF (*.gguf);;All files (*)")
            if p:
                sender = self.sender()
                if sender and sender.parentWidget():
                    # route by current empty field
                    if not self.model.text(): self.model.setText(p)
                    else: self.mmproj.setText(p)

        def _pick_exe(self):
            p, _ = QFileDialog.getOpenFileName(self, "Select executable", "", "Executable (*.exe);;All files (*)")
            if not p: return
            focused = self.focusWidget()
            if focused is self.server_exe: self.server_exe.setText(p)
            elif focused is self.pdflatex: self.pdflatex.setText(p)
            else: self.server_exe.setText(p)

        def _pick_dir(self):
            p = QFileDialog.getExistingDirectory(self, "Select folder")
            if p: self.output_dir.setText(p)

        def start_server(self):
            exe = self.server_exe.text().strip(); model = self.model.text().strip(); mm = self.mmproj.text().strip()
            if not exe or not model or not mm:
                QMessageBox.warning(self, APP_NAME, "Set llama-server.exe, model GGUF and mmproj GGUF first.")
                return
            if self.server.state() != QProcess.NotRunning:
                return
            args = ["-m", model, "--mmproj", mm, "--host", "127.0.0.1", "--port", str(self.port.value()), "-c", str(self.ctx.value()), "-ngl", str(self.ngl.value()), "--temp", str(self.temp.value()), "--top-p", str(self.top_p.value())]
            self.log_box.appendPlainText("Starting llama-server: " + " ".join([exe, *args]))
            self.server.start(exe, args)

        def stop_server(self):
            if self.server.state() != QProcess.NotRunning:
                self.server.terminate(); self.server.waitForFinished(3000)
                if self.server.state() != QProcess.NotRunning: self.server.kill()

        def analyze(self):
            if not self.image.text().strip():
                QMessageBox.warning(self, APP_NAME, "Select an image first."); return
            try:
                scene, context = analyze_image(self.image.text(), multiscale=self.multiscale.isChecked(), ocr="auto" if self.ocr.isChecked() else "off")
                self._scene = scene
                self.context_box.setPlainText(context)
                self.prompt_box.setPlainText(build_golden_prompt(scene))
                out = Path(self.output_dir.text().strip() or Path(self.image.text()).parent)
                out.mkdir(parents=True, exist_ok=True)
                (out / "scene.json").write_text(json.dumps(scene.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
                (out / "llm_context.txt").write_text(context, encoding="utf-8")
                (out / "golden_prompt.txt").write_text(build_golden_prompt(scene), encoding="utf-8")
                self.log_box.appendPlainText(f"Analysis complete. Outputs: {out}")
            except Exception as exc:
                self.log_box.appendPlainText(f"ERROR: {exc}")
                QMessageBox.critical(self, APP_NAME, str(exc))

        def copy_prompt(self):
            QApplication.clipboard().setText(self.prompt_box.toPlainText())
            self.log_box.appendPlainText("Golden Prompt copied to clipboard.")

        def verify(self):
            if not self.image.text().strip() or not self.tikz_box.toPlainText().strip():
                QMessageBox.warning(self, APP_NAME, "Provide the original image and generated TikZ first."); return
            result = compile_and_compare(self.image.text(), self.tikz_box.toPlainText(), pdflatex=self.pdflatex.text(), pdftoppm=self.pdftoppm.text())
            if result.score is None:
                msg = f"Compiled: {result.compiled}\nScore: unavailable\n{result.error or result.log[-3000:]}"
            else:
                msg = f"Compiled: {result.compiled}\nSimilarity score: {result.score:.4f}\nRendered: {result.rendered_image_path}"
            self.log_box.appendPlainText(msg)
            QMessageBox.information(self, APP_NAME, msg)

        def save(self):
            data = {
                "image": self.image.text(), "server_exe": self.server_exe.text(), "model": self.model.text(), "mmproj": self.mmproj.text(),
                "output_dir": self.output_dir.text(), "pdflatex": self.pdflatex.text(), "pdftoppm": self.pdftoppm.text(),
                "port": self.port.value(), "ctx": self.ctx.value(), "ngl": self.ngl.value(), "crops": self.crops.value(),
                "temp": self.temp.value(), "top_p": self.top_p.value(), "ocr": self.ocr.isChecked(), "multiscale": self.multiscale.isChecked(),
                "llm_endpoint": self.llm_endpoint.text(), "llm_model": self.llm_model.text(),
            }
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.log_box.appendPlainText(f"Settings saved: {CONFIG_PATH}")

        def _load(self):
            if not CONFIG_PATH.exists(): return
            try:
                d = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                for name in ("image", "server_exe", "model", "mmproj", "output_dir", "pdflatex", "pdftoppm", "llm_endpoint", "llm_model"):
                    if name in d: getattr(self, name).setText(str(d[name]))
                for name in ("port", "ctx", "ngl", "crops"):
                    if name in d: getattr(self, name).setValue(int(d[name]))
                for name in ("temp", "top_p"):
                    if name in d: getattr(self, name).setValue(float(d[name]))
                for name in ("ocr", "multiscale"):
                    if name in d: getattr(self, name).setChecked(bool(d[name]))
            except Exception as exc:
                self.log_box.appendPlainText(f"Could not load settings: {exc}")

    app = QApplication(sys.argv)
    window = Window(); window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
