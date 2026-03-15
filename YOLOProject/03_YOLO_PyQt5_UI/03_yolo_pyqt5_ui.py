# 导入系统模块，用于处理命令行参数和系统退出
import sys
# 导入 OpenCV 库，用于视频读取、图像处理和解码
import cv2
# 从 PyQt5.QtWidgets 导入构建图形界面所需的各种组件
# QMainWindow: 主窗口类，提供菜单栏、工具栏等框架
# QLabel: 标签控件，用于显示文本或图片
# QPushButton: 按钮控件
# QFileDialog: 文件选择对话框
# QVBoxLayout/HBoxLayout: 垂直/水平布局管理器，用于排列控件
# QWidget: 所有 UI 控件的基类
# QSlider: 滑动条控件
# QComboBox: 下拉选择框
# QMessageBox: 消息提示框（警告、错误、信息等）
# QGroupBox: 分组框，用于将相关控件归类显示
# QFormLayout: 表单布局，常用于“标签 - 输入框”成对排列
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QHBoxLayout, QWidget,
                             QSlider, QComboBox, QMessageBox, QGroupBox, QFormLayout)
# 从 PyQt5.QtCore 导入核心功能
# Qt: 包含各种枚举常量（如对齐方式、键盘键值等）
# QTimer: 定时器类（本代码主要用线程，但保留以备后用）
# QThread: 线程类，用于创建后台工作线程
# pyqtSignal: 信号类，用于线程间安全通信
# QObject: 所有 Qt 对象的基类
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
# 从 PyQt5.QtGui 导入图形相关类
# QImage: Qt 的图像格式，用于在 QLabel 中显示图片
# QPixmap: 用于高效渲染图像的类，通常由 QImage 转换而来
# QFont: 字体设置类
from PyQt5.QtGui import QImage, QPixmap, QFont
# 导入 Ultralytics YOLO 模型库
from ultralytics import YOLO
# 导入 os 模块，用于处理文件路径和检查文件是否存在
import os


# ---------------------------------------------------------
# 1. 工作线程类 (DetectionWorker)
# 作用：在后台线程执行耗时的 YOLO 推理，防止界面卡死
# ---------------------------------------------------------
class DetectionWorker(QThread):
    """
    专门用于运行 YOLO 检测的后台线程。
    继承自 QThread，重写了 run 方法作为线程入口。
    """

    # 定义自定义信号：frame_ready
    # 当检测到一帧图像时，发射此信号，携带处理后的 QImage 对象和统计信息字符串
    # 信号是线程安全的，允许子线程向主线程发送数据
    frame_ready = pyqtSignal(QImage, str)

    # 定义完成信号：finished
    # 当检测任务结束（视频播完或用户停止）时发射，通知主线程清理状态
    finished = pyqtSignal()

    def __init__(self, source, model, conf_threshold, classes=None):
        """
        初始化线程对象。
        :param source: 输入源（视频路径字符串、摄像头索引整数、或图片路径）
        :param model: 已加载的 YOLO 模型对象
        :param conf_threshold: 置信度阈值 (float, 0.0-1.0)
        :param classes: 要检测的类别 ID 列表 (例如 [0] 代表只检测人)，None 表示检测所有
        """
        # 调用父类 (QThread) 的构造函数
        super().__init__()

        self.source = source  # 保存输入源
        self.model = model  # 保存模型引用
        self.conf_threshold = conf_threshold  # 保存置信度阈值
        self.classes = classes  # 保存类别过滤列表
        self._stop_flag = False  # 定义停止标志位，用于安全终止线程循环

    def run(self):
        """
        线程启动时自动执行的方法。
        所有的耗时操作（视频读取、模型推理）都写在这里。
        """
        cap = None  # 初始化视频捕获对象变量
        is_image = False  # 标记当前处理的是否为静态图片

        # --- 第一步：判断输入源类型 ---

        # 如果 source 是字符串且是一个存在的文件
        if isinstance(self.source, str) and os.path.isfile(self.source):
            # 获取文件扩展名并转为小写
            ext = os.path.splitext(self.source)[1].lower()

            # 判断是否为图片格式
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                is_image = True  # 标记为图片模式

                # 使用 OpenCV 读取图片
                frame = cv2.imread(self.source)

                # 如果读取失败（文件损坏等），发射结束信号并退出线程
                if frame is None:
                    self.finished.emit()
                    return

                # 【图片模式】：只推理一次
                # model.predict: 执行检测，不启用追踪（因为是单张图）
                # verbose=False: 关闭控制台日志输出
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    classes=self.classes,
                    verbose=False
                )

                # 在图上绘制检测框和标签
                annotated_frame = results[0].plot()

                # 获取统计信息（如检测到的数量）
                count_info = self._get_stats(results[0])

                # 发送处理后的图片和信息到主界面
                self._send_frame(annotated_frame, count_info)

                # 图片处理完毕，发射结束信号
                self.finished.emit()
                return  # 退出 run 方法，线程结束

            else:
                # 如果不是图片，则认为是视频文件，打开视频捕获
                cap = cv2.VideoCapture(self.source)

        # 如果 source 是整数，认为是摄像头索引（如 0, 1, 2）
        elif isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source)

        # 检查视频/摄像头是否成功打开，且不是图片模式
        if not cap.isOpened() and not is_image:
            self.finished.emit()
            return

        # --- 第二步：视频/摄像头 循环处理 ---

        # 当停止标志位为 False 时，持续循环
        while not self._stop_flag:
            # 读取一帧视频
            # ret: 布尔值，True 表示读取成功
            # frame: 图像数据 (numpy 数组，BGR 格式)
            ret, frame = cap.read()

            # 如果读取失败（视频结束或摄像头断开），跳出循环
            if not ret:
                break

            # 【核心推理】执行 YOLO 追踪
            # stream=True: 启用生成器模式，避免一次性加载所有帧到内存，这对视频流至关重要
            # persist=True: 保持追踪器状态，确保跨帧的 ID 连续性
            results = self.model.track(
                source=frame,
                stream=True,
                conf=self.conf_threshold,
                classes=self.classes,
                tracker="bytetrack.yaml",  # 使用 ByteTrack 追踪算法
                persist=True,
                verbose=False
            )

            # 遍历结果生成器（通常每次只有一帧结果）
            for result in results:
                # 在帧上绘制检测框、ID 和类别
                annotated_frame = result.plot()

                # 获取当前帧的统计信息
                count_info = self._get_stats(result)

                # 发送信号：将处理好的帧和文本信息传给主线程
                self._send_frame(annotated_frame, count_info)

        # --- 第三步：清理资源 ---

        # 如果打开了视频或摄像头，释放资源
        if cap:
            cap.release()

        # 发射结束信号，通知主线程任务已完成
        self.finished.emit()

    def _send_frame(self, frame, info):
        """
        辅助函数：将 OpenCV 的 numpy 图像转换为 Qt 的 QImage 格式，并发射信号。
        :param frame: OpenCV 图像 (BGR 格式)
        :param info: 统计信息字符串
        """
        # OpenCV 默认使用 BGR 颜色空间，而 Qt 使用 RGB，必须转换否则颜色会偏蓝
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 获取图像的高、宽、通道数
        h, w, ch = rgb_frame.shape

        # 计算每行字节数：宽度 * 通道数 (每个通道 1 字节)
        bytes_per_line = ch * w

        # 创建 QImage 对象
        # 参数：数据指针，宽，高，每行字节数，像素格式 (RGB888)
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 发射信号，将 QImage 和文本传递给连接该信号的槽函数
        self.frame_ready.emit(q_image, info)

    def _get_stats(self, result):
        """
        辅助函数：解析检测结果，生成统计字符串。
        :param result: 单帧的 YOLO 结果对象
        :return: 统计信息字符串
        """
        # 如果是 track 模式，检查是否有追踪 ID
        if result.boxes.id is not None:
            # 获取所有 ID 并转为 numpy 数组
            ids = result.boxes.id.cpu().numpy()
            # 计算唯一 ID 的数量（去重后的总人数）
            unique_ids = len(set(ids))
            # 计算当前帧检测框的总数
            total_boxes = len(ids)
            return f"追踪ID数：{unique_ids} | 当前框数：{total_boxes}"

        # 如果是 detect 模式（无 ID），只统计框数
        elif result.boxes is not None and len(result.boxes) > 0:
            return f"当前框数：{len(result.boxes)}"

        # 如果没有检测到任何目标
        else:
            return "未检测到目标"

    def stop(self):
        """
        外部调用的停止方法。
        设置停止标志位，并等待线程安全退出。
        """
        self._stop_flag = True  # 设置标志位，让 run 方法中的 while 循环在下一次判断时退出
        self.wait()  # 阻塞当前调用线程，直到工作线程完全结束


# ---------------------------------------------------------
# 2. 主窗口类 (YOLODetectorApp)
# 作用：构建 GUI 界面，处理用户交互，管理线程生命周期
# ---------------------------------------------------------
class YOLODetectorApp(QMainWindow):
    def __init__(self):
        """
        主窗口构造函数，初始化界面和默认状态。
        """
        super().__init__()  # 调用父类 QMainWindow 的构造函数

        self.model = None  # 用于存储加载的 YOLO 模型
        self.worker = None  # 用于存储当前的工作线程对象
        self.timer = QTimer()  # 预留定时器（本例主要靠线程信号，暂未使用）

        self.init_ui()  # 调用方法初始化 UI 布局
        # 默认加载最轻量级的 Nano 模型
        self.load_model('yolov8n.pt')

    def init_ui(self):
        """
        初始化界面布局：创建控件、设置属性、添加事件监听。
        """
        # 设置窗口标题
        self.setWindowTitle("YOLOv8 智能目标检测工具 (PyQt5)")
        # 设置窗口初始位置和大小 (x, y, width, height)
        self.setGeometry(100, 100, 1000, 700)

        # 创建一个中央容器 widget，所有控件都将放在这里面
        central_widget = QWidget()
        # 将中央容器设置为窗口的中心部件
        self.setCentralWidget(central_widget)

        # 创建水平布局：左侧控制面板，右侧显示区域
        main_layout = QHBoxLayout(central_widget)

        # ================= 左侧：控制面板 =================
        control_panel = QWidget()
        # 固定左侧面板宽度为 250 像素
        control_panel.setFixedWidth(250)
        # 创建垂直布局，控件从上到下排列
        control_layout = QVBoxLayout(control_panel)

        # --- 1. 模型选择区域 ---
        model_group = QGroupBox("模型选择")  # 创建带标题的分组框
        model_layout = QFormLayout()  # 使用表单布局（标签在左，控件在右）

        self.model_combo = QComboBox()  # 创建下拉框
        # 添加可选的模型名称
        self.model_combo.addItems(['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'])
        # 连接信号：当下拉框内容改变时，触发 on_model_change 方法
        self.model_combo.currentTextChanged.connect(self.on_model_change)

        # 将标签和下拉框添加到表单布局
        model_layout.addRow("模型:", self.model_combo)
        model_group.setLayout(model_layout)  # 将布局应用到分组框

        # --- 2. 检测设置区域 ---
        conf_group = QGroupBox("检测设置")
        conf_layout = QFormLayout()

        # 创建滑动条：水平方向
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(10)  # 最小值 10 (代表 0.10)
        self.conf_slider.setMaximum(90)  # 最大值 90 (代表 0.90)
        self.conf_slider.setValue(40)  # 默认值 40 (代表 0.40)
        # 连接信号：滑块值改变时，更新旁边的标签文字
        self.conf_slider.valueChanged.connect(self.update_conf_label)

        # 创建标签显示当前数值
        self.conf_label = QLabel("0.40")

        # 创建类别选择下拉框
        self.class_check = QComboBox()
        self.class_check.addItems(['全部类别', '仅人员 (Person)'])

        # 添加到布局
        conf_layout.addRow("置信度:", self.conf_slider)
        conf_layout.addRow("", self.conf_label)  # 第一列为空，第二列放标签
        conf_layout.addRow("目标类别:", self.class_check)

        conf_group.setLayout(conf_layout)

        # --- 3. 操作按钮区域 ---
        btn_group = QGroupBox("操作")
        btn_layout = QVBoxLayout()  # 垂直排列按钮

        # 创建“打开图片”按钮
        self.btn_load_img = QPushButton("📷 打开图片")
        # 连接点击事件
        self.btn_load_img.clicked.connect(self.load_image)

        # 创建“打开视频”按钮
        self.btn_load_vid = QPushButton("🎬 打开视频")
        self.btn_load_vid.clicked.connect(self.load_video)

        # 创建“开启摄像头”按钮
        self.btn_camera = QPushButton("📹 开启摄像头")
        self.btn_camera.clicked.connect(self.start_camera)

        # 创建“停止”按钮
        self.btn_stop = QPushButton("⏹ 停止/重置")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False)  # 初始状态禁用，因为没有任务在运行

        # 将按钮加入布局
        btn_layout.addWidget(self.btn_load_img)
        btn_layout.addWidget(self.btn_load_vid)
        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_stop)
        btn_group.setLayout(btn_layout)

        # --- 4. 状态信息显示 ---
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)  # 文字居中
        # 设置样式表：灰色斜体
        self.status_label.setStyleSheet("color: gray; font-style: italic;")

        # 将所有左侧组件加入主布局
        control_layout.addWidget(model_group)
        control_layout.addWidget(conf_group)
        control_layout.addWidget(btn_group)
        control_layout.addStretch()  # 添加弹性空间，把状态栏挤到底部
        control_layout.addWidget(self.status_label)

        # ================= 右侧：显示区域 =================
        display_group = QGroupBox("检测预览")
        display_layout = QVBoxLayout()

        # 创建用于显示图像的标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)  # 图片居中
        self.image_label.setMinimumSize(640, 480)  # 设置最小尺寸
        # 设置背景色为黑色，文字白色（当没有图片时显示提示文字）
        self.image_label.setStyleSheet("background-color: #000000; color: white;")
        self.image_label.setText("暂无图像")

        # 创建用于显示统计信息的标签
        self.info_label = QLabel("等待输入...")
        self.info_label.setAlignment(Qt.AlignCenter)
        # 设置字体加粗，绿色文字，黑色背景
        self.info_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.info_label.setStyleSheet("color: #00FF00; background-color: black; padding: 5px;")

        # 组装右侧布局
        display_layout.addWidget(self.image_label)
        display_layout.addWidget(self.info_label)
        display_group.setLayout(display_layout)

        # --- 最终组装：将左右两部分加入主水平布局 ---
        main_layout.addWidget(control_panel)  # 添加左侧
        main_layout.addWidget(display_group)  # 添加右侧

    def load_model(self, model_name):
        """
        加载指定的 YOLO 模型。
        :param model_name: 模型文件名
        """
        # 更新状态栏提示用户正在加载
        self.status_label.setText(f"正在加载模型：{model_name} ...")
        # 强制刷新 UI 事件队列，确保用户能立刻看到“正在加载”的文字，而不是等加载完才变
        QApplication.processEvents()

        try:
            # 实例化 YOLO 模型，会自动下载或加载本地权重
            self.model = YOLO(model_name)
            self.status_label.setText(f"模型已加载：{model_name}")
        except Exception as e:
            # 如果加载失败（如网络问题），弹出错误框
            QMessageBox.critical(self, "错误", f"模型加载失败:\n{e}")
            self.status_label.setText("模型加载失败")

    def on_model_change(self, model_name):
        """
        当用户在 ComboBox 中切换模型时触发。
        """
        # 先停止当前正在运行的检测任务
        self.stop_detection()
        # 加载新选择的模型
        self.load_model(model_name)

    def update_conf_label(self, value):
        """
        当滑块拖动时，更新旁边标签显示的数值。
        :param value: 滑块当前的整数值 (10-90)
        """
        conf = value / 100.0  # 转换为小数
        self.conf_label.setText(f"{conf:.2f}")  # 格式化为两位小数

    def get_current_classes(self):
        """
        根据下拉框选择，返回需要检测的类别 ID 列表。
        :return: list 或 None
        """
        # 如果用户选择了“仅人员”
        if self.class_check.currentText() == '仅人员 (Person)':
            return [0]  # COCO 数据集中，0 代表 'person'
        return None  # 返回 None 表示检测所有类别

    def stop_detection(self):
        """
        停止当前的检测任务，释放资源，重置按钮状态。
        """
        # 如果存在正在运行的工作线程
        if self.worker and self.worker.isRunning():
            self.worker.stop()  # 调用线程的 stop 方法（设置标志位并等待）
            self.worker.wait()  # 确保线程完全退出
            self.worker = None  # 清空引用

        # 恢复按钮状态
        self.btn_stop.setEnabled(False)  # 禁用停止按钮
        self.btn_load_img.setEnabled(True)  # 启用图片按钮
        self.btn_load_vid.setEnabled(True)  # 启用视频按钮
        self.btn_camera.setEnabled(True)  # 启用摄像头按钮

        self.status_label.setText("已停止")
        self.info_label.setText("等待输入...")
        # 注意：这里没有清除 image_label 的图片，保留最后一帧供用户查看

    def start_worker(self, source):
        """
        通用方法：创建并启动检测线程。
        :param source: 输入源（路径或索引）
        """
        # 检查模型是否已加载
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先等待模型加载完成！")
            return

        # 启动新任务前，确保旧任务已停止
        self.stop_detection()

        # 获取当前设置的置信度 (滑块值 / 100)
        conf = self.conf_slider.value() / 100.0
        # 获取当前选择的类别
        classes = self.get_current_classes()

        # 实例化工作线程
        self.worker = DetectionWorker(source, self.model, conf, classes)

        # 【关键】连接信号与槽
        # 当线程发出 frame_ready 信号时，调用本类的 update_display 方法更新 UI
        self.worker.frame_ready.connect(self.update_display)
        # 当线程发出 finished 信号时，调用 on_worker_finished 进行清理
        self.worker.finished.connect(self.on_worker_finished)

        # 启动线程（会自动调用 run 方法）
        self.worker.start()

        # 更新按钮状态：检测中禁用输入按钮，启用停止按钮
        self.btn_stop.setEnabled(True)
        self.btn_load_img.setEnabled(False)
        self.btn_load_vid.setEnabled(False)
        self.btn_camera.setEnabled(False)
        self.status_label.setText("检测进行中...")

    def update_display(self, q_image, info_text):
        """
        槽函数：接收线程发来的图像和信息，更新界面。
        此方法运行在主线程（UI 线程）中。
        :param q_image: Qt 图像对象
        :param info_text: 统计信息字符串
        """
        # 将 QImage 转换为 QPixmap（Qt 用于显示的高效格式）
        pixmap = QPixmap.fromImage(q_image)

        # 缩放图片以适应 label 的大小，同时保持宽高比
        # Qt.KeepAspectRatio: 保持比例
        # Qt.SmoothTransformation: 使用平滑算法缩放，避免锯齿
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 将处理好的图片设置到 label 上显示
        self.image_label.setPixmap(scaled_pixmap)

        # 更新下方的统计信息文本
        self.info_label.setText(info_text)

    def on_worker_finished(self):
        """
        槽函数：当检测线程自然结束（如视频播完）时调用。
        """
        self.status_label.setText("任务完成")
        self.btn_stop.setEnabled(False)
        # 恢复输入按钮，允许用户进行下一次操作
        self.btn_load_img.setEnabled(True)
        self.btn_load_vid.setEnabled(True)
        self.btn_camera.setEnabled(True)
        # 注意：worker 对象这里没有置为 None，方便调试，但逻辑上任务已结束

    # --- 按钮点击事件的具体实现 ---

    def load_image(self):
        """打开文件对话框选择图片"""
        # 获取文件路径，过滤器限制只显示图片格式
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        # 如果用户选择了文件（file_path 不为空）
        if file_path:
            self.start_worker(file_path)

    def load_video(self):
        """打开文件对话框选择视频"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.start_worker(file_path)

    def start_camera(self):
        """启动默认摄像头"""
        # 这里直接硬编码使用摄像头索引 0
        # 进阶做法可以弹出一个 QInputDialog 让用户输入索引
        self.start_worker(0)


# ---------------------------------------------------------
# 3. 程序入口
# ---------------------------------------------------------
if __name__ == '__main__':
    # 启用高 DPI 缩放支持，确保在高分屏（如 Retina）上界面不模糊
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # 启用高 DPI 图标支持
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # 创建 QApplication 实例，sys.argv 允许通过命令行传递参数
    app = QApplication(sys.argv)

    # 实例化主窗口
    window = YOLODetectorApp()
    # 显示窗口
    window.show()

    # 进入应用程序的主事件循环
    # exec_() 会一直运行，直到窗口关闭或程序退出
    # sys.exit() 确保程序退出时返回正确的状态码
    sys.exit(app.exec_())
