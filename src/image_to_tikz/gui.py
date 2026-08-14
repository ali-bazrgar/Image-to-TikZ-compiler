from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path

from .golden_prompt import build_golden_prompt
from .llama_semantic import enrich_scene_with_llama_server_vlm
from .llama_server_vlm import LlamaServerVLMError, LlamaServerVisionObserver, build_llama_server_command
from .llm_client import chat_completion
from .pipeline import analyze_image
from .tikz_verifier import compile_and_compare

APP_NAME = "Image-to-TikZ Studio"
CONFIG_PATH = Path(os.environ.get("APPDATA", Path.home())) / "ImageToTikZ" / "settings.json"


def run() -> int:
    try:
        from PySide6.QtCore import QProcess
        from PySide6.QtWidgets import (
            QApplication, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
            QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QPlainTextEdit,
            QSpinBox, QDoubleSpinBox, QTabWidget, QVBoxLayout, QWidget, QCheckBox,
        )
    except ImportError as exc:
        raise SystemExit("GUI requires optional dependency: pip install -e '.[gui]'") from exc

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(APP_NAME)
            self.resize(1240, 860)
            self.server = QProcess(self)
            self.scene = None
            self._build()
            self._load()

        def _build(self) -> None:
            root = QWidget(); self.setCentralWidget(root)
            outer = QVBoxLayout(root)

            files = QGroupBox("1. Files")
            grid = QGridLayout(files)
            self.image = QLineEdit(); self.server_exe = QLineEdit(); self.model = QLineEdit(); self.mmproj = QLineEdit()
            self.output_dir = QLineEdit(); self.pdflatex = QLineEdit("pdflatex"); self.pdftoppm = QLineEdit("pdftoppm")
            self._file_row(grid, 0, "Image", self.image, self._pick_image)
            self._file_row(grid, 1, "llama-server", self.server_exe, partial(self._pick_exe, self.server_exe))
            self._file_row(grid, 2, "SmolVLM2 Q4_K_M", self.model, partial(self._pick_file, self.model))
            self._file_row(grid, 3, "SmolVLM2 mmproj", self.mmproj, partial(self._pick_file, self.mmproj))
            self._file_row(grid, 4, "Output folder", self.output_dir, self._pick_dir)
            self._file_row(grid, 5, "pdflatex", self.pdflatex, partial(self._pick_exe, self.pdflatex))
            self._file_row(grid, 6, "pdftoppm", self.pdftoppm, partial(self._pick_exe, self.pdftoppm))
            outer.addWidget(files)

            vision = QGroupBox("2. Vision / llama.cpp")
            form = QFormLayout(vision)
            self.port = QSpinBox(); self.port.setRange(1, 65535); self.port.setValue(8080)
            self.ctx = QSpinBox(); self.ctx.setRange(512, 32768); self.ctx.setValue(4096)
            self.ngl = QSpinBox(); self.ngl.setRange(0, 200); self.ngl.setValue(99)
            self.crops = QSpinBox(); self.crops.setRange(0, 32); self.crops.setValue(8)
            self.smol_temp = QDoubleSpinBox(); self.smol_temp.setRange(0.0, 2.0); self.smol_temp.setSingleStep(0.05); self.smol_temp.setValue(0.1)
            self.smol_top_p = QDoubleSpinBox(); self.smol_top_p.setRange(0.01, 1.0); self.smol_top_p.setSingleStep(0.01); self.smol_top_p.setValue(0.9)
            self.ocr = QCheckBox("Enable lightweight OCR")
            self.multiscale = QCheckBox("Multiscale CV analysis"); self.multiscale.setChecked(True)
            self.no_mmproj_offload = QCheckBox("Keep mmproj on CPU (useful if VRAM is tight)")
            form.addRow("llama port", self.port); form.addRow("context", self.ctx); form.addRow("GPU layers", self.ngl)
            form.addRow("Semantic crops", self.crops); form.addRow("Smol temperature", self.smol_temp); form.addRow("Smol top-p", self.smol_top_p)
            form.addRow(self.no_mmproj_offload); form.addRow(self.ocr); form.addRow(self.multiscale)
            outer.addWidget(vision)

            llm = QGroupBox("3. Commercial LLM (OpenAI-compatible)")
            lform = QFormLayout(llm)
            self.llm_endpoint = QLineEdit("https://api.openai.com/v1")
            self.llm_model = QLineEdit(); self.llm_key = QLineEdit(); self.llm_key.setEchoMode(QLineEdit.Password)
            lform.addRow("Endpoint", self.llm_endpoint); lform.addRow("Model", self.llm_model); lform.addRow("API key (not saved)", self.llm_key)
            outer.addWidget(llm)

            buttons = QHBoxLayout()
            for text, slot in [
                ("Start llama.cpp", self.start_server), ("Check Vision", self.check_vision), ("Stop", self.stop_server),
                ("Analyze + Smol", self.analyze), ("Send to LLM", self.send_to_llm), ("Copy Prompt", self.copy_prompt),
                ("Verify TikZ", self.verify), ("Save settings", self.save),
            ]:
                b = QPushButton(text); b.clicked.connect(slot); buttons.addWidget(b)
            outer.addLayout(buttons)

            self.tabs = QTabWidget()
            self.context_box = QPlainTextEdit(); self.context_box.setReadOnly(True)
            self.prompt_box = QPlainTextEdit(); self.prompt_box.setReadOnly(True)
            self.tikz_box = QPlainTextEdit(); self.log_box = QPlainTextEdit(); self.log_box.setReadOnly(True)
            self.tabs.addTab(self.context_box, "Visual Context"); self.tabs.addTab(self.prompt_box, "Golden Prompt")
            self.tabs.addTab(self.tikz_box, "TikZ"); self.tabs.addTab(self.log_box, "Log")
            outer.addWidget(self.tabs, 1)
            self.server.readyReadStandardOutput.connect(self._drain_server_stdout)
            self.server.readyReadStandardError.connect(self._drain_server_stderr)

        @staticmethod
        def _file_row(grid, row, label, edit, callback):
            grid.addWidget(QLabel(label), row, 0); grid.addWidget(edit, row, 1)
            b = QPushButton("Browse"); b.clicked.connect(callback); grid.addWidget(b, row, 2)

        def _pick_image(self):
            p, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
            if p: self.image.setText(p)

        def _pick_file(self, edit):
            p, _ = QFileDialog.getOpenFileName(self, "Select GGUF", "", "GGUF (*.gguf);;All files (*)")
            if p: edit.setText(p)

        def _pick_exe(self, edit):
            p, _ = QFileDialog.getOpenFileName(self, "Select executable", "", "Executable (*.exe);;All files (*)")
            if p: edit.setText(p)

        def _pick_dir(self):
            p = QFileDialog.getExistingDirectory(self, "Select output folder")
            if p: self.output_dir.setText(p)

        def start_server(self):
            exe, model, mm = self.server_exe.text().strip(), self.model.text().strip(), self.mmproj.text().strip()
            if not all((exe, model, mm)):
                QMessageBox.warning(self, APP_NAME, "Select llama-server.exe, SmolVLM2 GGUF and mmproj GGUF first."); return
            if self.server.state() != QProcess.NotRunning: return
            args = build_llama_server_command(
                exe, model, mm,
                host="127.0.0.1", port=self.port.value(), context=self.ctx.value(), gpu_layers=self.ngl.value(),
                no_mmproj_offload=self.no_mmproj_offload.isChecked(),
            )
            args += ["--temp", str(self.smol_temp.value()), "--top-p", str(self.smol_top_p.value())]
            self.log_box.appendPlainText("START: " + " ".join([exe, *args])); self.server.start(exe, args)

        def stop_server(self):
            if self.server.state() != QProcess.NotRunning:
                self.server.terminate()
                if not self.server.waitForFinished(3000): self.server.kill()

        def _server_url(self) -> str:
            return f"http://127.0.0.1:{self.port.value()}/v1"

        def check_vision(self):
            try:
                if self.server.state() == QProcess.NotRunning:
                    raise LlamaServerVLMError("llama-server is not running. Click 'Start llama.cpp' first.")
                observer = LlamaServerVisionObserver(
                    self.model.text().strip(), self.mmproj.text().strip(),
                    base_url=self._server_url(), max_model_bytes=2_500_000_000,
                )
                props = observer.check_server()
                self.log_box.appendPlainText(
                    "VISION OK: modalities=" + json.dumps(props.get("modalities", {}), ensure_ascii=False)
                    + "; build=" + str(props.get("build_info", "unknown"))
                )
                QMessageBox.information(self, APP_NAME, "Vision is enabled on the llama.cpp server.")
            except Exception as exc:
                self.log_box.appendPlainText("VISION CHECK FAILED: " + str(exc))
                QMessageBox.critical(self, APP_NAME, str(exc))

        def analyze(self):
            image = self.image.text().strip()
            if not image:
                QMessageBox.warning(self, APP_NAME, "Select an image first."); return
            try:
                scene, _ = analyze_image(image, multiscale=self.multiscale.isChecked(), ocr="auto" if self.ocr.isChecked() else "off")
                if self.server.state() != QProcess.NotRunning and self.model.text().strip() and self.mmproj.text().strip() and self.crops.value() > 0:
                    scene = enrich_scene_with_llama_server_vlm(scene, image, self.model.text().strip(), self.mmproj.text().strip(), base_url=self._server_url(), max_crops=self.crops.value(), max_model_bytes=2_500_000_000)
                    self.log_box.appendPlainText("SmolVLM2 semantic crop analysis complete.")
                else:
                    self.log_box.appendPlainText("SmolVLM2 skipped: start llama.cpp and configure model/mmproj to enable semantic inspection.")
                from .serialize import to_llm_context
                context = to_llm_context(scene); self.scene = scene
                self.context_box.setPlainText(context); self.prompt_box.setPlainText(build_golden_prompt(scene))
                out = Path(self.output_dir.text().strip() or Path(image).parent); out.mkdir(parents=True, exist_ok=True)
                (out / "scene.json").write_text(json.dumps(scene.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
                (out / "llm_context.txt").write_text(context, encoding="utf-8")
                (out / "golden_prompt.txt").write_text(build_golden_prompt(scene), encoding="utf-8")
                self.log_box.appendPlainText(f"ANALYSIS COMPLETE: {out}")
            except Exception as exc:
                self.log_box.appendPlainText(f"ERROR: {exc}"); QMessageBox.critical(self, APP_NAME, str(exc))

        def send_to_llm(self):
            prompt, endpoint, model = self.prompt_box.toPlainText().strip(), self.llm_endpoint.text().strip(), self.llm_model.text().strip()
            if not prompt or not endpoint or not model:
                QMessageBox.warning(self, APP_NAME, "Analyze first and set commercial LLM endpoint + model."); return
            try:
                result = chat_completion(endpoint, model, prompt, api_key=self.llm_key.text().strip(), temperature=0.0, max_tokens=12000)
                self.tikz_box.setPlainText(result); self.tabs.setCurrentWidget(self.tikz_box)
                out = Path(self.output_dir.text().strip() or Path(self.image.text()).parent); out.mkdir(parents=True, exist_ok=True)
                (out / "tikz_response.txt").write_text(result, encoding="utf-8")
                self.log_box.appendPlainText("Commercial LLM response received.")
            except Exception as exc:
                self.log_box.appendPlainText(f"LLM ERROR: {exc}"); QMessageBox.critical(self, APP_NAME, str(exc))

        def copy_prompt(self):
            QApplication.clipboard().setText(self.prompt_box.toPlainText()); self.log_box.appendPlainText("Golden Prompt copied.")

        def verify(self):
            image, tikz = self.image.text().strip(), self.tikz_box.toPlainText().strip()
            if not image or not tikz:
                QMessageBox.warning(self, APP_NAME, "Provide the original image and generated TikZ first."); return
            result = compile_and_compare(image, tikz, pdflatex=self.pdflatex.text(), pdftoppm=self.pdftoppm.text())
            msg = f"Compiled: {result.compiled}\n"
            if result.score is not None: msg += f"Similarity: {result.score:.4f}\n"
            if result.error: msg += result.error
            elif result.rendered_image_path: msg += f"Rendered: {result.rendered_image_path}"
            self.log_box.appendPlainText(msg); QMessageBox.information(self, APP_NAME, msg)

        def _drain_server_stdout(self): self.log_box.appendPlainText(bytes(self.server.readAllStandardOutput()).decode(errors="replace"))
        def _drain_server_stderr(self): self.log_box.appendPlainText(bytes(self.server.readAllStandardError()).decode(errors="replace"))

        def save(self):
            data = {
                "image": self.image.text(), "server_exe": self.server_exe.text(), "model": self.model.text(), "mmproj": self.mmproj.text(),
                "output_dir": self.output_dir.text(), "pdflatex": self.pdflatex.text(), "pdftoppm": self.pdftoppm.text(),
                "port": self.port.value(), "ctx": self.ctx.value(), "ngl": self.ngl.value(), "crops": self.crops.value(),
                "smol_temp": self.smol_temp.value(), "smol_top_p": self.smol_top_p.value(), "no_mmproj_offload": self.no_mmproj_offload.isChecked(),
                "ocr": self.ocr.isChecked(), "multiscale": self.multiscale.isChecked(),
                "llm_endpoint": self.llm_endpoint.text(), "llm_model": self.llm_model.text(),
            }
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True); CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.log_box.appendPlainText(f"SETTINGS SAVED: {CONFIG_PATH}")

        def _load(self):
            if not CONFIG_PATH.exists(): return
            try:
                d = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                for name in ("image", "server_exe", "model", "mmproj", "output_dir", "pdflatex", "pdftoppm", "llm_endpoint", "llm_model"):
                    if name in d: getattr(self, name).setText(str(d[name]))
                for name in ("port", "ctx", "ngl", "crops"):
                    if name in d: getattr(self, name).setValue(int(d[name]))
                for name in ("smol_temp", "smol_top_p"):
                    if name in d: getattr(self, name).setValue(float(d[name]))
                self.no_mmproj_offload.setChecked(bool(d.get("no_mmproj_offload", False)))
                self.ocr.setChecked(bool(d.get("ocr", False))); self.multiscale.setChecked(bool(d.get("multiscale", True)))
            except Exception as exc: self.log_box.appendPlainText(f"SETTINGS LOAD ERROR: {exc}")

    app = QApplication(sys.argv); window = Window(); window.show(); return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
