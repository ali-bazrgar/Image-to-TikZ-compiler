from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .golden_prompt import build_golden_prompt
from .llama_server_vlm import LlamaServerVisionObserver, enrich_scene_with_llama_server_vlm, build_llama_server_command
from .llm_client import chat_completion
from .pipeline import analyze_image
from .tikz_verifier import compile_and_compare

APP_NAME = "Image-to-TikZ Studio"
CONFIG_PATH = Path(os.environ.get("APPDATA", str(Path.home()))) / "ImageToTikZ" / "settings.json"


def run() -> int:
    try:
        from PySide6.QtCore import QObject, QProcess, QThread, Signal, Slot
        from PySide6.QtWidgets import QApplication, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QPlainTextEdit, QProgressBar, QSpinBox, QDoubleSpinBox, QTabWidget, QVBoxLayout, QWidget, QCheckBox
    except ImportError as exc:
        raise SystemExit("GUI requires: pip install -e '.[gui]'") from exc

    class AnalysisWorker(QObject):
        finished = Signal(object, str, str)
        failed = Signal(str)
        progress = Signal(str)

        def __init__(self, image: str, multiscale: bool, ocr_enabled: bool, server_url: str, model: str, mmproj: str, crops: int):
            super().__init__(); self.image=image; self.multiscale=multiscale; self.ocr_enabled=ocr_enabled; self.server_url=server_url; self.model=model; self.mmproj=mmproj; self.crops=crops

        @Slot()
        def run(self):
            try:
                self.progress.emit("Running deterministic visual analysis...")
                scene, _ = analyze_image(self.image, multiscale=self.multiscale, ocr="auto" if self.ocr_enabled else "off")
                if self.model and self.mmproj and self.crops > 0:
                    self.progress.emit("Checking llama.cpp vision endpoint...")
                    observer = LlamaServerVisionObserver(self.model, self.mmproj, base_url=self.server_url, max_model_bytes=2_500_000_000)
                    observer.check_server()
                    self.progress.emit("Analyzing selected semantic crops with SmolVLM2...")
                    scene = enrich_scene_with_llama_server_vlm(scene, self.image, self.model, self.mmproj, base_url=self.server_url, max_crops=self.crops, max_model_bytes=2_500_000_000)
                else:
                    self.progress.emit("SmolVLM2 skipped; continuing with deterministic context.")
                from .serialize import to_llm_context
                context = to_llm_context(scene); prompt = build_golden_prompt(scene); self.finished.emit(scene, context, prompt)
            except Exception as exc:
                self.failed.emit(f"{type(exc).__name__}: {exc}")

    class Window(QMainWindow):
        def __init__(self):
            super().__init__(); self.setWindowTitle(APP_NAME); self.resize(1240, 860)
            self.server=QProcess(self); self.analysis_thread=None; self.analysis_worker=None; self.scene=None; self._build(); self._load()
            self.server.started.connect(lambda: self._log("llama-server process started.")); self.server.errorOccurred.connect(lambda e: self._log(f"llama-server QProcess error: {e}")); self.server.finished.connect(lambda c,s: self._log(f"llama-server exited: code={c}, status={s}")); self.server.readyReadStandardOutput.connect(self._drain_server_stdout); self.server.readyReadStandardError.connect(self._drain_server_stderr)

        def _build(self):
            root=QWidget(); self.setCentralWidget(root); outer=QVBoxLayout(root); files=QGroupBox("1. Files"); grid=QGridLayout(files)
            self.image=QLineEdit(); self.server_exe=QLineEdit(); self.model=QLineEdit(); self.mmproj=QLineEdit(); self.output_dir=QLineEdit(); self.pdflatex=QLineEdit("pdflatex"); self.pdftoppm=QLineEdit("pdftoppm")
            rows=[("Image",self.image,self._pick_image),("llama-server.exe",self.server_exe,lambda:self._pick_exe(self.server_exe)),("SmolVLM2 Q4_K_M",self.model,lambda:self._pick_file(self.model)),("SmolVLM2 mmproj",self.mmproj,lambda:self._pick_file(self.mmproj)),("Output folder",self.output_dir,self._pick_dir),("pdflatex",self.pdflatex,lambda:self._pick_exe(self.pdflatex)),("pdftoppm",self.pdftoppm,lambda:self._pick_exe(self.pdftoppm))]
            for r,(label,edit,cb) in enumerate(rows): grid.addWidget(QLabel(label),r,0); grid.addWidget(edit,r,1); b=QPushButton("Browse"); b.clicked.connect(cb); grid.addWidget(b,r,2)
            outer.addWidget(files)
            vision=QGroupBox("2. Vision / llama.cpp"); form=QFormLayout(vision)
            self.port=QSpinBox(); self.port.setRange(1,65535); self.port.setValue(8080); self.ctx=QSpinBox(); self.ctx.setRange(512,32768); self.ctx.setValue(4096); self.ngl=QSpinBox(); self.ngl.setRange(0,200); self.ngl.setValue(99); self.parallel=QSpinBox(); self.parallel.setRange(1,8); self.parallel.setValue(1); self.crops=QSpinBox(); self.crops.setRange(0,32); self.crops.setValue(8); self.smol_temp=QDoubleSpinBox(); self.smol_temp.setRange(0,2); self.smol_temp.setSingleStep(.05); self.smol_temp.setValue(.1); self.smol_top_p=QDoubleSpinBox(); self.smol_top_p.setRange(.01,1); self.smol_top_p.setSingleStep(.01); self.smol_top_p.setValue(.9); self.auto_fit=QCheckBox("Auto-fit GPU layers"); self.auto_fit.setChecked(True); self.no_mmproj_offload=QCheckBox("Keep mmproj on CPU"); self.ocr=QCheckBox("Enable lightweight OCR"); self.multiscale=QCheckBox("Multiscale CV analysis"); self.multiscale.setChecked(True)
            form.addRow("Port",self.port); form.addRow("Context",self.ctx); form.addRow("GPU layers",self.ngl); form.addRow("Server slots",self.parallel); form.addRow("Semantic crops",self.crops); form.addRow("Smol temperature",self.smol_temp); form.addRow("Smol top-p",self.smol_top_p); form.addRow(self.auto_fit); form.addRow(self.no_mmproj_offload); form.addRow(self.ocr); form.addRow(self.multiscale); outer.addWidget(vision)
            llm=QGroupBox("3. Commercial LLM (OpenAI-compatible)"); lf=QFormLayout(llm); self.llm_endpoint=QLineEdit("https://api.openai.com/v1"); self.llm_model=QLineEdit(); self.llm_key=QLineEdit(); self.llm_key.setEchoMode(QLineEdit.Password); lf.addRow("Endpoint",self.llm_endpoint); lf.addRow("Model",self.llm_model); lf.addRow("API key (not saved)",self.llm_key); outer.addWidget(llm)
            buttons=QHBoxLayout(); self.start_btn=QPushButton("Start llama.cpp"); self.check_btn=QPushButton("Check Vision"); self.stop_btn=QPushButton("Stop"); self.analyze_btn=QPushButton("Analyze + Smol"); self.send_btn=QPushButton("Send to LLM"); self.verify_btn=QPushButton("Verify TikZ"); self.save_btn=QPushButton("Save settings")
            for b,fn in ((self.start_btn,self.start_server),(self.check_btn,self.check_vision),(self.stop_btn,self.stop_server),(self.analyze_btn,self.analyze),(self.send_btn,self.send_to_llm),(self.verify_btn,self.verify),(self.save_btn,self.save)): b.clicked.connect(fn); buttons.addWidget(b)
            outer.addLayout(buttons); self.progress=QProgressBar(); self.progress.setRange(0,0); self.progress.hide(); outer.addWidget(self.progress)
            self.tabs=QTabWidget(); self.context_box=QPlainTextEdit(); self.context_box.setReadOnly(True); self.prompt_box=QPlainTextEdit(); self.prompt_box.setReadOnly(True); self.tikz_box=QPlainTextEdit(); self.log_box=QPlainTextEdit(); self.log_box.setReadOnly(True); self.tabs.addTab(self.context_box,"Visual Context"); self.tabs.addTab(self.prompt_box,"Golden Prompt"); self.tabs.addTab(self.tikz_box,"TikZ"); self.tabs.addTab(self.log_box,"Log"); outer.addWidget(self.tabs,1)

        def _log(self,text): self.log_box.appendPlainText(text)
        def _pick_image(self): p,_=QFileDialog.getOpenFileName(self,"Select image","","Images (*.png *.jpg *.jpeg *.webp *.bmp)"); self.image.setText(p) if p else None
        def _pick_file(self,edit): p,_=QFileDialog.getOpenFileName(self,"Select GGUF","","GGUF (*.gguf);;All files (*)"); edit.setText(p) if p else None
        def _pick_exe(self,edit): p,_=QFileDialog.getOpenFileName(self,"Select executable","","Executable (*.exe);;All files (*)"); edit.setText(p) if p else None
        def _pick_dir(self): p=QFileDialog.getExistingDirectory(self,"Select output folder"); self.output_dir.setText(p) if p else None
        def _server_url(self): return f"http://127.0.0.1:{self.port.value()}/v1"
        def _set_busy(self,busy):
            for b in (self.start_btn,self.check_btn,self.analyze_btn,self.send_btn,self.verify_btn,self.save_btn): b.setEnabled(not busy)
            self.progress.setVisible(busy)
        def start_server(self):
            exe,model,mm=self.server_exe.text().strip(),self.model.text().strip(),self.mmproj.text().strip()
            if not all(Path(p).is_file() for p in (exe,model,mm)): QMessageBox.warning(self,APP_NAME,"Select valid llama-server.exe, model GGUF and mmproj GGUF."); return
            if self.server.state()!=QProcess.NotRunning: self._log("llama-server is already running."); return
            args=build_llama_server_command(exe,model,mm,host="127.0.0.1",port=self.port.value(),context=self.ctx.value(),gpu_layers=self.ngl.value())
            if self.auto_fit.isChecked() and "-ngl" in args: i=args.index("-ngl"); del args[i:i+2]; args += ["--fit","on","--fit-target","384"]
            args += ["--parallel",str(self.parallel.value()),"--temp",str(self.smol_temp.value()),"--top-p",str(self.smol_top_p.value())]
            if self.no_mmproj_offload.isChecked(): args.append("--no-mmproj-offload")
            self.server.setWorkingDirectory(str(Path(exe).resolve().parent)); self._log("START COMMAND:"); self._log("  "+" ".join([exe,*args])); self.server.start(exe,args)
        def stop_server(self):
            if self.server.state()!=QProcess.NotRunning: self.server.terminate(); self.server.waitForFinished(3000)
        def check_vision(self):
            if self.server.state()==QProcess.NotRunning: QMessageBox.warning(self,APP_NAME,"Start llama.cpp first."); return
            try: props=LlamaServerVisionObserver(self.model.text(),self.mmproj.text(),base_url=self._server_url(),max_model_bytes=2_500_000_000).check_server(); self._log("VISION PROPS: "+json.dumps(props.get("modalities",{}),ensure_ascii=False)); QMessageBox.information(self,APP_NAME,"Vision is enabled on llama.cpp.")
            except Exception as exc: self._log("VISION CHECK FAILED: "+str(exc)); QMessageBox.critical(self,APP_NAME,str(exc))
        def analyze(self):
            if self.analysis_thread and self.analysis_thread.isRunning(): return
            image=self.image.text().strip()
            if not image: QMessageBox.warning(self,APP_NAME,"Select an image first."); return
            if self.crops.value()>0 and self.server.state()==QProcess.NotRunning: QMessageBox.warning(self,APP_NAME,"Start llama.cpp first when SmolVLM2 semantic analysis is enabled."); return
            self._set_busy(True); self._log("ANALYSIS STARTED..."); self.analysis_thread=QThread(self); self.analysis_worker=AnalysisWorker(image,self.multiscale.isChecked(),self.ocr.isChecked(),self._server_url(),self.model.text().strip(),self.mmproj.text().strip(),self.crops.value()); self.analysis_worker.moveToThread(self.analysis_thread); self.analysis_thread.started.connect(self.analysis_worker.run); self.analysis_worker.progress.connect(self._log); self.analysis_worker.finished.connect(self._analysis_finished); self.analysis_worker.failed.connect(self._analysis_failed); self.analysis_worker.finished.connect(self.analysis_thread.quit); self.analysis_worker.failed.connect(self.analysis_thread.quit); self.analysis_thread.finished.connect(self._analysis_thread_finished); self.analysis_thread.start()
        @Slot(object,str,str)
        def _analysis_finished(self,scene,context,prompt):
            self.scene=scene; self.context_box.setPlainText(context); self.prompt_box.setPlainText(prompt); out=Path(self.output_dir.text().strip() or Path(self.image.text()).parent); out.mkdir(parents=True,exist_ok=True); (out/"scene.json").write_text(json.dumps(scene.to_dict(),ensure_ascii=False,indent=2),encoding="utf-8"); (out/"llm_context.txt").write_text(context,encoding="utf-8"); (out/"golden_prompt.txt").write_text(prompt,encoding="utf-8"); self._log(f"ANALYSIS COMPLETE: {out}")
        @Slot(str)
        def _analysis_failed(self,msg): self._log("ANALYSIS FAILED: "+msg); QMessageBox.critical(self,APP_NAME,"Analysis failed:\n"+msg)
        def _analysis_thread_finished(self): self._set_busy(False); self.analysis_worker=None; self.analysis_thread.deleteLater(); self.analysis_thread=None
        def send_to_llm(self):
            prompt,endpoint,model=self.prompt_box.toPlainText().strip(),self.llm_endpoint.text().strip(),self.llm_model.text().strip()
            if not prompt or not endpoint or not model: QMessageBox.warning(self,APP_NAME,"Analyze first and set endpoint + model."); return
            try: self.tikz_box.setPlainText(chat_completion(endpoint,model,prompt,api_key=self.llm_key.text().strip(),temperature=0.0,max_tokens=12000)); self.tabs.setCurrentWidget(self.tikz_box)
            except Exception as exc: self._log("LLM ERROR: "+str(exc)); QMessageBox.critical(self,APP_NAME,str(exc))
        def verify(self):
            try: r=compile_and_compare(self.image.text().strip(),self.tikz_box.toPlainText().strip(),pdflatex=self.pdflatex.text(),pdftoppm=self.pdftoppm.text()); msg=f"Compiled: {r.compiled}\n"+(f"Similarity: {r.score:.4f}\n" if r.score is not None else "")+(r.error or ""); self._log(msg); QMessageBox.information(self,APP_NAME,msg)
            except Exception as exc: self._log("VERIFY ERROR: "+str(exc)); QMessageBox.critical(self,APP_NAME,str(exc))
        def save(self):
            data={"image":self.image.text(),"server_exe":self.server_exe.text(),"model":self.model.text(),"mmproj":self.mmproj.text(),"output_dir":self.output_dir.text(),"pdflatex":self.pdflatex.text(),"pdftoppm":self.pdftoppm.text(),"port":self.port.value(),"ctx":self.ctx.value(),"ngl":self.ngl.value(),"parallel":self.parallel.value(),"crops":self.crops.value(),"auto_fit":self.auto_fit.isChecked(),"smol_temp":self.smol_temp.value(),"smol_top_p":self.smol_top_p.value(),"no_mmproj_offload":self.no_mmproj_offload.isChecked(),"ocr":self.ocr.isChecked(),"multiscale":self.multiscale.isChecked(),"llm_endpoint":self.llm_endpoint.text(),"llm_model":self.llm_model.text()}; CONFIG_PATH.parent.mkdir(parents=True,exist_ok=True); CONFIG_PATH.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8"); self._log("SETTINGS SAVED: "+str(CONFIG_PATH))
        def _load(self):
            if not CONFIG_PATH.exists(): return
            try:
                d=json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                for n in ("image","server_exe","model","mmproj","output_dir","pdflatex","pdftoppm","llm_endpoint","llm_model"):
                    if n in d: getattr(self,n).setText(str(d[n]))
                for n in ("port","ctx","ngl","parallel","crops"):
                    if n in d: getattr(self,n).setValue(int(d[n]))
                for n in ("smol_temp","smol_top_p"):
                    if n in d: getattr(self,n).setValue(float(d[n]))
                self.auto_fit.setChecked(bool(d.get("auto_fit",True))); self.no_mmproj_offload.setChecked(bool(d.get("no_mmproj_offload",False))); self.ocr.setChecked(bool(d.get("ocr",False))); self.multiscale.setChecked(bool(d.get("multiscale",True)))
            except Exception as exc: self._log("SETTINGS LOAD ERROR: "+str(exc))
        def _drain_server_stdout(self): self._log(bytes(self.server.readAllStandardOutput()).decode(errors="replace"))
        def _drain_server_stderr(self): self._log(bytes(self.server.readAllStandardError()).decode(errors="replace"))
        def closeEvent(self,event):
            if self.analysis_thread and self.analysis_thread.isRunning(): QMessageBox.warning(self,APP_NAME,"Analysis is still running; wait for completion before closing."); event.ignore(); return
            self.stop_server(); event.accept()

    app=QApplication(sys.argv); w=Window(); w.show(); return app.exec()


if __name__ == "__main__": raise SystemExit(run())
