# 导入必要的库
# cv2: OpenCV 库，用于视频读取、图像处理、窗口显示和视频保存
import cv2
# os: 操作系统接口库，用于检查文件是否存在、处理路径等
import os
# YOLO: 从 ultralytics 库导入 YOLO 模型类，用于目标检测和追踪
from ultralytics import YOLO


def track_people_in_video(video_path, output_path="output_result.mp4"):
    """
    对视频中的人物进行检测和追踪，包含异常处理机制。
    :param video_path: 输入视频路径 (字符串) 或 摄像头索引 (整数，如 0)
    :param output_path: 输出结果视频的保存路径
    """
    # 初始化变量为 None，防止在 try 块出错前未定义导致 finally 块报错
    cap = None  # 用于存储视频捕获对象 (摄像头或视频文件)
    out = None  # 用于存储视频写入对象 (用于保存结果)

    try:
        # ---------------------------------------------------------
        # 1. 加载 AI 模型
        # ---------------------------------------------------------
        print("正在加载 YOLOv8 模型...")
        # 实例化 YOLO 模型，自动下载或使用本地的 'yolov8n.pt' (Nano 版本，速度最快)
        # 如果第一次运行，它会自动从网络下载模型权重文件
        model = YOLO('yolov8n.pt')

        # 获取模型包含的所有类别名称 (例如: {0: 'person', 1: 'bicycle', ...})
        class_names = model.names

        # 检查 'person' 是否在模型支持的类别中 (COCO 数据集通常都有)
        if 'person' not in class_names.values():
            print("错误：模型中未找到 'person' 类别")
            return  # 如果没有人类类别，直接退出函数

        # 获取 'person' 对应的类别 ID (在 COCO 数据集中通常是 0)
        # 逻辑：先找到 'person' 在 values 中的索引，再用该索引去 keys 中取 ID
        person_class_id = list(class_names.keys())[list(class_names.values()).index('person')]

        # ---------------------------------------------------------
        # 2. 打开视频源 (文件或摄像头)
        # ---------------------------------------------------------
        # 判断输入是文件路径还是摄像头索引
        # 如果是整数 (如 0)，认为是摄像头；如果是字符串且文件不存在，报错
        is_camera = isinstance(video_path, int)

        if not is_camera and not os.path.exists(video_path):
            print(f"错误：找不到视频文件 '{video_path}'")
            return

        # 创建 VideoCapture 对象
        # 参数可以是文件路径 (str) 或 摄像头索引 (int, 0 代表默认摄像头)
        cap = cv2.VideoCapture(video_path)

        # 检查是否成功打开视频源
        if not cap.isOpened():
            print("错误：无法打开视频文件或摄像头")
            return

        # ---------------------------------------------------------
        # 3. 获取视频属性并配置写入器
        # ---------------------------------------------------------
        # 获取视频帧的宽度 (像素)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 获取视频帧的高度 (像素)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取视频帧率 (FPS, 每秒传输帧数)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 容错处理：如果获取不到 FPS (返回 0.0)，默认设为 25.0，防止后续写入报错
        if fps <= 0:
            fps = 25.0

        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 设置进度显示标志位
        show_progress_percent = True
        # 如果总帧数为 0 (常见于摄像头或某些编码格式)，则不显示百分比
        if total_frames <= 0:
            print("警告：无法获取视频总帧数 (可能是摄像头或特殊编码)，将仅显示当前帧数。")
            show_progress_percent = False
        else:
            # 打印视频基本信息
            print(f"视频信息：{frame_width}x{frame_height}, FPS: {fps}, 总帧数: {total_frames}")

        # 配置视频编码器 (FourCC code)
        # 'avc1' 是 H.264 编码，在 macOS 和大多数现代播放器上兼容性最好
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

        # 创建 VideoWriter 对象，用于将处理后的帧保存为新视频
        # 参数：输出路径，编码器，帧率，画面尺寸 (宽, 高)
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # 检查输出文件是否成功创建
        if not out.isOpened():
            print("错误：无法创建输出视频文件，请检查编码器支持或磁盘权限。")
            return

        print(f"开始处理... (按 'q' 键退出预览，或 Ctrl+C 停止)")

        # 初始化帧计数器
        frame_count = 0
        # 初始化一个集合 (set)，用于存储所有出现过的唯一人物 ID (自动去重)
        unique_track_ids = set()

        # ---------------------------------------------------------
        # 4. 主循环：逐帧读取、推理、绘制、保存
        # ---------------------------------------------------------
        while True:
            # 读取下一帧
            # ret: 布尔值，表示是否成功读取帧
            # frame: 图像数据 (numpy 数组)
            ret, frame = cap.read()

            # 如果读取失败 (例如视频结束)，跳出循环
            if not ret:
                print("视频播放结束。")
                break

            # 帧计数器加 1
            frame_count += 1

            # 每处理 10 帧打印一次进度，避免刷屏太快
            if frame_count % 10 == 0:
                if show_progress_percent:
                    # 计算百分比进度
                    percent = (frame_count / total_frames) * 100
                    # end='\r' 表示打印后不换行，而是回到行首覆盖，形成进度条效果
                    print(f"进度: {frame_count}/{total_frames} ({percent:.1f}%)", end='\r')
                else:
                    print(f"正在处理第 {frame_count} 帧...", end='\r')

            # 【核心步骤】执行 YOLO 追踪推理
            results = model.track(
                source=frame,  # 输入当前帧图像
                stream=True,  # 启用生成器模式，节省内存 (处理视频必选)
                classes=[person_class_id],  # 只检测 'person' 类，忽略其他物体，提高速度
                tracker="bytetrack.yaml",  # 使用 ByteTrack 算法进行多目标追踪 (保持 ID 连续)
                persist=True,  # 保持追踪状态跨帧 (如果不加，每帧 ID 都会重置)
                verbose=False,  # 关闭模型自带的详细日志输出
                conf=0.4,  # 置信度阈值：只有确信度>0.4 的框才被保留
                iou=0.5  # IoU 阈值：用于非极大值抑制 (NMS)，去除重叠框
            )

            # 遍历推理结果 (因为 stream=True，results 是一个生成器)
            for result in results:
                # 检查是否检测到了框，并且是否有追踪 ID (track 模式下才有 id)
                if result.boxes.id is not None:
                    # 提取追踪 ID 并转换为 numpy 数组
                    track_ids = result.boxes.id.cpu().numpy()

                    # 将当前帧检测到的所有 ID 加入集合 (自动去重)
                    for tid in track_ids:
                        unique_track_ids.add(int(tid))

                    # 在原图上绘制检测框、ID 和类别标签
                    # line_width: 线宽, font_size: 字体大小
                    annotated_frame = result.plot(line_width=2, font_size=12)

                    # 在画面左上角额外绘制当前帧检测到的人数
                    # 参数：图像，文字，位置 (x,y)，字体，缩放比例，颜色 (BGR), 线宽
                    cv2.putText(annotated_frame, f"People: {len(track_ids)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    # 如果没检测到人，直接使用原图，不画框
                    annotated_frame = frame

                # 在窗口中显示处理后的图像
                # 窗口名称："Video Person Tracking"
                cv2.imshow("Video Person Tracking", annotated_frame)

                # 将处理后的帧写入输出视频文件
                out.write(annotated_frame)

                # 检测键盘输入
                # waitKey(1): 等待 1 毫秒，& 0xFF 是为了兼容不同系统
                # ord('q'): 获取字符 'q' 的 ASCII 码
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户按下 'q' 键，正在停止...")
                    break  # 跳出 while 循环

        # ---------------------------------------------------------
        # 5. 循环结束后的统计与清理
        # ---------------------------------------------------------
        print("\n" + "-" * 30)
        print("处理完成！")
        print(f"输出文件已保存：{output_path}")
        # 打印集合的大小，即视频中出现的不同人物的总数
        print(f"检测到不同人物总数 (Unique IDs): {len(unique_track_ids)}")
        print("-" * 30)

    except KeyboardInterrupt:
        # 捕获用户按下 Ctrl+C 的中断信号
        print("\n\n[系统中断] 检测到用户强制停止 (Ctrl+C)。")
        print("正在安全清理资源...")

    except Exception as e:
        # 捕获其他所有未知异常
        print(f"\n[发生未知错误]: {e}")

    finally:
        # 【关键】finally 块中的代码无论是否出错都会执行
        # 确保资源被正确释放，防止文件损坏或摄像头被占用

        # 释放视频捕获对象 (关闭摄像头或视频文件)
        if cap is not None:
            cap.release()

        # 释放视频写入对象 (非常重要：不 release 会导致输出视频文件损坏或无法播放)
        if out is not None:
            out.release()

        # 关闭所有 OpenCV 创建的窗口
        cv2.destroyAllWindows()
        print("资源已释放，程序退出。")


if __name__ == '__main__':
    # 配置输入源
    # 选项 A: 使用本地视频文件 -> video_file = "test_video.mp4"
    # 选项 B: 使用摄像头 -> video_file = 0

    video_file = 0  # 这里设置为 0，表示调用默认摄像头

    # 逻辑判断：如果是整数 (摄像头)，跳过文件存在性检查；如果是字符串，检查文件是否存在
    if isinstance(video_file, int):
        print(f"提示：已设置为使用摄像头 (索引：{video_file})。请确保摄像头可用。")
        track_people_in_video(video_file)
    elif not os.path.exists(video_file):
        # 如果是字符串路径且文件不存在
        print(f"提示：未找到文件 '{video_file}'，请放入视频文件或修改路径。")
    else:
        # 文件存在，开始处理
        track_people_in_video(video_file)
