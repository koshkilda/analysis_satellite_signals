#!/usr/bin/env python3
"""
Интерфейс GNSS Analysis Studio
Запускает анализ данных и последующую визуализацию графиков.
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QStatusBar, QMessageBox, 
                            QProgressBar, QFrame, QGraphicsDropShadowEffect, QDialog)
from PyQt5.QtCore import Qt, QProcess, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def setup_qt_environment():
    """Настройка переменных окружения для корректной работы Qt."""
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    if "XDG_SESSION_TYPE" in os.environ:
        del os.environ["XDG_SESSION_TYPE"]
    os.environ["QT_XCB_FORCE_SOFTWARE_OPENGL"] = "1"
    os.environ["QT_DEBUG_PLUGINS"] = "0"

class ModernButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4e73df,stop:1 #224abe);
                color:white; border:none; border-radius:12px; font-weight:bold; font-size:14px;
                padding:12px 24px; min-height:45px; min-width:200px; }
            QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6a89e6,stop:1 #3a5bd8); }
            QPushButton:pressed { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #3a5bd8,stop:1 #224abe);
                padding-top:14px; padding-bottom:10px; }
            QPushButton:disabled { background:#9ca3af; }
        """)
        self.setCursor(Qt.PointingHandCursor)

class PlotCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setStyleSheet("QFrame{background:white;border-radius:16px;border:1px solid #e2e8f0;}")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0,0,0,30))
        shadow.setOffset(0,4)
        self.setGraphicsEffect(shadow)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,15)
        layout.setSpacing(10)
        lbl = QLabel(title)
        lbl.setFont(QFont("Arial", 14, QFont.Bold))
        lbl.setStyleSheet("color:#1e3a8a;")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        self.plot_container = QWidget()
        self.plot_container.setStyleSheet("background:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;")
        layout.addWidget(self.plot_container, 1)
        self.status = QLabel("График будет здесь")
        self.status.setStyleSheet("color:#64748b;font-style:italic;")
        self.status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status)

class GNSSAnalyzerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GNSS Analysis Studio")
        self.setGeometry(100, 100, 1600, 1000)
        self.setup_style()
        
        cw = QWidget()
        self.setCentralWidget(cw)
        main = QVBoxLayout(cw)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(15)
        
        # Заголовок
        hl = QHBoxLayout()
        hl.setContentsMargins(0, 0, 0, 15)
        tl = QVBoxLayout()
        t = QLabel("GNSS ANALYSIS STUDIO")
        t.setFont(QFont("Arial", 24, QFont.Bold))
        t.setStyleSheet("color:#1e3a8a;")
        t.setAlignment(Qt.AlignCenter)
        tl.addWidget(t)
        hl.addLayout(tl, 1)
        main.addLayout(hl)
        
        # Управление
        cl = QHBoxLayout()
        cl.setSpacing(15)
        self.load_btn = ModernButton("ЗАГРУЗИТЬ ДАННЫЕ")
        self.load_btn.setMinimumHeight(50)
        self.load_btn.setFont(QFont("Arial", 14, QFont.Bold))
        self.load_btn.clicked.connect(self.load_data)
        self.status_lbl = QLabel("Выберите CSV файл")
        self.status_lbl.setFont(QFont("Arial", 12))
        self.status_lbl.setStyleSheet("color:#64748b;font-style:italic;padding:10px;background:#f1f5f9;border-radius:10px;border:1px solid #cbd5e1;")
        self.status_lbl.setWordWrap(True)
        cl.addWidget(self.load_btn, 1)
        cl.addWidget(self.status_lbl, 2)
        main.addLayout(cl)
        
        # Панель стратегии
        self.setup_strategy_panel(main)
        
        # Прогресс
        self.pb = QProgressBar()
        self.pb.setVisible(False)
        self.pb.setStyleSheet("QProgressBar{border:none;border-radius:10px;background:#e2e8f0;height:10px;}QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #3b82f6,stop:1 #63b3ed);border-radius:10px;}")
        main.addWidget(self.pb)
        
        # Графики
        gl = QHBoxLayout()
        gl.setSpacing(20)
        self.p2d = PlotCard("2D график")
        gl.addWidget(self.p2d, 1)
        self.p3d = PlotCard("3D график")
        gl.addWidget(self.p3d, 1)
        main.addLayout(gl, 1)
        
        # Статус-бар
        self.sb = QStatusBar()
        self.setStatusBar(self.sb)
        self.sb.setStyleSheet("QStatusBar{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #f8fafc,stop:1 #e2e8f0);border-top:1px solid #cbd5e1;color:#4b5563;font-size:13px;height:30px;}")
        self.sb.showMessage("Готов", 3000)
        self.ind = QLabel()
        self.ind.setFixedSize(15, 15)
        self.ind.setStyleSheet("background:#10b981;border-radius:7px;")
        self.sb.addPermanentWidget(self.ind)
        
        self.setup_plots()
        
        # Процесс C++ анализатора
        self.cpp_proc = QProcess()
        self.cpp_proc.setProcessChannelMode(QProcess.MergedChannels)
        self.cpp_proc.readyReadStandardOutput.connect(self.on_cpp_output)
        self.cpp_proc.finished.connect(self.on_cpp_finished)
        
        # Процесс визуализации Python
        self.viz_proc = QProcess()
        self.viz_proc.setProcessChannelMode(QProcess.MergedChannels)
        self.viz_proc.readyReadStandardOutput.connect(self.on_viz_output)
        self.viz_proc.finished.connect(self.on_viz_finished)
        
        self.ptimer = QTimer()
        self.ptimer.timeout.connect(self.on_progress)
        self.pval = 0
        
        # Пути
        self.curdir = Path(__file__).parent.absolute()
        self.cpp_path = self.curdir / "analysis_app"
        self.viz_path = self.curdir / "plot_generator.py" 
        self.resdir = self.curdir / "results"
        self.resdir.mkdir(exist_ok=True)
        
        self.out_unified = self.resdir / "data_unified.csv"
        self.p2d_path = self.resdir / "2d_plot.png"
        self.p3d_path = self.resdir / "3d_plot.png"
        self.metrics_plot_path = self.resdir / "metrics_plot.png"
        # Удалена строка self.metrics_btn = None – кнопка уже создана в setup_strategy_panel

    def setup_style(self):
        """Применяет палитру и стили интерфейса."""
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(248, 250, 252))
        pal.setColor(QPalette.WindowText, QColor(30, 41, 59))
        pal.setColor(QPalette.Base, QColor(255, 255, 255))
        pal.setColor(QPalette.Highlight, QColor(59, 130, 246))
        self.setPalette(pal)
        self.setStyleSheet("""
            QMainWindow{background:#f8fafc;} QLabel,QPushButton{font-family:'Segoe UI',Arial,sans-serif;}
            QPushButton[checkable="true"]{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #f1f5f9,stop:1 #e2e8f0);color:#475569;border:2px solid #cbd5e1;border-radius:10px;font-weight:bold;font-size:13px;padding:8px 12px;min-height:38px;}
            QPushButton[checkable="true"]:checked{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4e73df,stop:1 #224abe);color:white;border-color:#3b82f6;}
            QPushButton[checkable="true"]:hover{border-color:#60a5fa;background:#f8fafc;}
            QPushButton[checkable="true"]:checked:hover{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #5b84e8,stop:1 #2f58c4);}
            QPushButton[checkable="true"]:disabled{background:#e2e8f0;color:#9ca3af;border-color:#cbd5e1;}
        """)

    def setup_strategy_panel(self, parent):
        """Создаёт панель выбора стратегии анализа."""
        cf = QFrame()
        cf.setStyleSheet("QFrame{background:#fff;border-radius:12px;border:1px solid #e2e8f0;}")
        lay = QHBoxLayout(cf)
        lay.setContentsMargins(15, 10, 15, 10)
        lay.setSpacing(15)
        lbl = QLabel("Стратегия:")
        lbl.setFont(QFont("Arial", 13, QFont.Bold))
        lbl.setStyleSheet("color:#1e3a8a;")
        lay.addWidget(lbl)
        self.strats = ["AutoKMeans", "DBSCAN", "EM-GMM"]
        self.sbtns = []
        for s in self.strats:
            b = QPushButton(s)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(self.on_strat_click)
            self.sbtns.append(b)
            lay.addWidget(b, 1)
        self.sbtns[0].setChecked(True)
        self.sel_strat = self.strats[0]
        lay.addStretch(1)
        
        # Кнопка фильтрации полностью удалена (закомментирована)
        # self.fbtn = QPushButton("Фильтрация: ВКЛ")
        # self.fbtn.setCheckable(True)
        # self.fbtn.setChecked(True)
        # self.fbtn.setCursor(Qt.PointingHandCursor)
        # self.fbtn.clicked.connect(self.on_filt_toggle)
        # lay.addWidget(self.fbtn)
        # self.filt_on = True
        self.filt_on = False   # фильтрация отключена
        
        # Новая кнопка для показа графика метрик
        self.metrics_btn = QPushButton("📊 Показать графики метрик")
        self.metrics_btn.setCursor(Qt.PointingHandCursor)
        self.metrics_btn.setEnabled(False)   # станет активной после успешного анализа
        self.metrics_btn.clicked.connect(self.show_metrics_plot)
        self.metrics_btn.setStyleSheet("""
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4e73df,stop:1 #224abe);
                color:white; border:none; border-radius:12px; font-weight:bold; font-size:13px;
                padding:8px 16px; min-height:38px; }
            QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6a89e6,stop:1 #3a5bd8); }
            QPushButton:pressed { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #3a5bd8,stop:1 #224abe); }
            QPushButton:disabled { background:#9ca3af; }
        """)
        lay.addWidget(self.metrics_btn)
        
        parent.addWidget(cf)

    def on_strat_click(self):
        """Обработка нажатия кнопок стратегии."""
        s = self.sender()
        if s.isChecked():
            self.sel_strat = s.text()
            for b in self.sbtns:
                if b != s: b.setChecked(False)
        else:
            s.setChecked(True)

    def setup_plots(self):
        """Инициализирует контейнеры для графиков."""
        self.f2d = Figure(figsize=(10, 8), dpi=120)
        self.c2d = FigureCanvas(self.f2d)
        self.c2d.setStyleSheet("background:transparent;")
        l2 = QVBoxLayout(self.p2d.plot_container)
        l2.setContentsMargins(0, 0, 0, 0)
        l2.addWidget(self.c2d)
        
        self.f3d = Figure(figsize=(15, 8), dpi=120)
        self.c3d = FigureCanvas(self.f3d)
        self.c3d.setStyleSheet("background:transparent;")
        l3 = QVBoxLayout(self.p3d.plot_container)
        l3.setContentsMargins(0, 0, 0, 0)
        l3.addWidget(self.c3d)
        
        self.show_empty(self.f2d, self.c2d, "2D график появится после загрузки")
        self.show_empty(self.f3d, self.c3d, "3D график появится после загрузки")

    def show_empty(self, fig, cvs, msg):
        """Отображает заглушку вместо графика."""
        fig.clear()
        ax = fig.add_subplot(111)
        g = np.linspace(0, 1, 256)
        g = np.vstack((g, g))
        ax.imshow(g, aspect='auto', cmap='Blues', alpha=0.2)
        ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=14, color='#4b5563', 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cbd5e1', boxstyle='round,pad=1'))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        cvs.draw()

    def load_data(self):
        """Открывает диалог выбора файла и запускает анализ."""
        fp, _ = QFileDialog.getOpenFileName(self, "Выберите CSV", str(Path.home()), "CSV Files (*.csv);;All Files (*)")
        if not fp:
            return
        self.status_lbl.setText(f"Файл: {Path(fp).name}")
        self.sb.showMessage(f"Анализ: {Path(fp).name}", 3000)
        self.ind.setStyleSheet("background:#f59e0b;border-radius:7px;")
        self.run_analysis(fp)

    def run_analysis(self, fp):
        """Запускает C++ анализатор."""
        if not self.cpp_path.exists():
            QMessageBox.critical(self, "Ошибка", f"Не найден анализатор:\n{self.cpp_path}")
            return
        if not self.viz_path.exists():
            QMessageBox.critical(self, "Ошибка", f"Не найден скрипт визуализации:\n{self.viz_path}\nПереименуйте или измените путь в коде.")
            return
            
        self.load_btn.setEnabled(False)
        for b in self.sbtns:
            b.setEnabled(False)
        self.metrics_btn.setEnabled(False)   # на время анализа отключаем
        self.pb.setVisible(True)
        self.pb.setValue(0)
        self.pval = 0
        self.ptimer.start(40)
        
        # Очистка старых результатов
        for f in [self.out_unified, self.p2d_path, self.p3d_path, self.metrics_plot_path]:
            if f.exists():
                f.unlink()
        
        method_map = {"EM-GMM": "em", "DBSCAN": "kde-dbscan", "AutoKMeans": "kde-kmeans"}
        args = [fp, "--method", method_map.get(self.sel_strat, "em"), "--output-dir", str(self.resdir)]
        # Фильтрация отключена – не добавляем --filter
        
        self.status_lbl.setText(f"Стратегия: {self.sel_strat} | Фильтр: ВЫКЛ")
        self.sb.showMessage("Запуск анализа...", 2000)
        print(f"[UI] Запуск анализатора: {self.cpp_path} {' '.join(args)}")
        
        self.cpp_proc.start(str(self.cpp_path), args)
        if not self.cpp_proc.waitForStarted(3000):
            self.err(f"Не удалось запустить анализатор:\n{self.cpp_proc.errorString()}")

    def on_progress(self):
        """Обновляет полосу прогресса."""
        self.pval += 0.5
        if self.pval > 99:
            self.pval = 99
        self.pb.setValue(int(self.pval))

    def on_cpp_output(self):
        """Вывод C++ процесса в консоль для отладки."""
        d = self.cpp_proc.readAllStandardOutput().data().decode('utf-8', errors='ignore')
        if d.strip():
            print(f"[C++] {d.strip()}")

    def on_cpp_finished(self, code, status):
        """Обработка завершения C++ анализа. Запускает визуализацию."""
        self.ptimer.stop()
        self.pb.setValue(100)
        
        if code != 0 or status != QProcess.NormalExit:
            print(f"[C++] Завершён с ошибкой (код {code}, статус {status})")
            if not self.out_unified.exists():
                self.err("Анализатор не создал файл данных. Проверьте входной CSV и параметры.")
                return
        
        print("[UI] Анализ завершён. Запуск генерации графиков...")
        self.status_lbl.setText("Генерация графиков...")
        self.sb.showMessage("Создание графиков...", 2000)
        
        # Явный запуск визуализации через системный Python
        python_exe = sys.executable
        viz_args = [
            str(self.viz_path),
            str(self.out_unified),
            "--output-dir", str(self.resdir),
            "--verbose"
        ]
        # Добавляем metrics.csv, если существует (plot_generator.py сам его найдёт)
        metrics_csv = self.resdir / "metrics.csv"
        if metrics_csv.exists():
            viz_args.extend(["--metrics-csv", str(metrics_csv)])
        
        print(f"[UI] Запуск визуализатора: {python_exe} {' '.join(viz_args)}")
        self.viz_proc.start(python_exe, viz_args)
        if not self.viz_proc.waitForStarted(2000):
            self.err(f"Не удалось запустить визуализатор:\n{self.viz_proc.errorString()}")

    def on_viz_output(self):
        """Вывод Python-скрипта в консоль."""
        d = self.viz_proc.readAllStandardOutput().data().decode('utf-8', errors='ignore')
        if d.strip():
            print(f"[VIZ] {d.strip()}")

    def on_viz_finished(self, code, status):
        """Обработка завершения визуализации и отображение графиков."""
        if code != 0 or status != QProcess.NormalExit:
            print(f"[VIZ] Завершён с ошибкой (код {code})")
            self.err("Ошибка генерации графиков. Проверьте консоль вывода.")
            return
            
        print("[UI] Графики успешно созданы.")
        self.status_lbl.setText("Готово!")
        self.sb.showMessage("Успех", 3000)
        self.ind.setStyleSheet("background:#10b981;border-radius:7px;")
        self.show_results()
        
        # Активируем кнопку метрик, если файл существует
        if self.metrics_plot_path.exists():
            self.metrics_btn.setEnabled(True)
        
        self.load_btn.setEnabled(True)
        for b in self.sbtns:
            b.setEnabled(True)
        QTimer.singleShot(1500, lambda: self.pb.setVisible(False))

    def err(self, msg):
        """Отображает ошибку."""
        self.load_btn.setEnabled(True)
        for b in self.sbtns:
            b.setEnabled(True)
        self.metrics_btn.setEnabled(False)
        self.ptimer.stop()
        self.pb.setVisible(False)
        self.ind.setStyleSheet("background:#ef4444;border-radius:7px;")
        QMessageBox.critical(self, "Ошибка", msg[:500])
        self.status_lbl.setText("Ошибка")
        print(f"[UI] ОШИБКА: {msg}")

    def show_results(self):
        """Загружает и отображает сохранённые графики."""
        self.plot2d()
        self.plot3d()

    def plot2d(self):
        """Отображает 2D график."""
        try:
            if not self.p2d_path.exists():
                self.p2d.status.setText("Нет графика")
                return
            self.f2d.clear()
            ax = self.f2d.add_subplot(111)
            ax.axis('off')
            ax.imshow(plt.imread(str(self.p2d_path)))
            self.f2d.tight_layout()
            self.c2d.draw()
            self.p2d.status.setText("2D загружен")
            self.p2d.status.setStyleSheet("color:#10b981;font-weight:bold;")
        except Exception as e:
            print(f"[UI] Ошибка отрисовки 2D: {e}")

    def plot3d(self):
        """Отображает 3D график."""
        try:
            if not self.p3d_path.exists():
                self.p3d.status.setText("Нет графика")
                return
            self.f3d.clear()
            ax = self.f3d.add_subplot(111)
            ax.axis('off')
            ax.imshow(plt.imread(str(self.p3d_path)))
            self.f3d.tight_layout()
            self.c3d.draw()
            self.p3d.status.setText("3D загружен")
            self.p3d.status.setStyleSheet("color:#10b981;font-weight:bold;")
        except Exception as e:
            print(f"[UI] Ошибка отрисовки 3D: {e}")

    def show_metrics_plot(self):
        """Отображает окно с графиком метрик."""
        if not self.metrics_plot_path.exists():
            QMessageBox.warning(self, "Нет данных", "Файл metrics_plot.png не найден. Запустите анализ заново.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Графики метрик кластеризации")
        dlg.resize(1000, 800)
        layout = QVBoxLayout(dlg)
        label = QLabel()
        pixmap = QPixmap(str(self.metrics_plot_path))
        # Масштабируем под размер окна, сохраняя пропорции
        scaled = pixmap.scaled(dlg.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        layout.addWidget(label)
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close)
        dlg.exec_()

    def closeEvent(self, ev):
        """Корректное завершение работы приложения."""
        if self.cpp_proc.state() == QProcess.Running:
            self.cpp_proc.terminate()
            self.cpp_proc.waitForFinished(3000)
        if self.viz_proc.state() == QProcess.Running:
            self.viz_proc.terminate()
            self.viz_proc.waitForFinished(3000)
        ev.accept()

if __name__ == "__main__":
    setup_qt_environment()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = GNSSAnalyzerUI()
    w.show()
    sys.exit(app.exec_())