import cv2
import os
from ultralytics import YOLO


def track_people_in_video(video_path, output_path="output_result.mp4"):
    """
    对视频中的人物进行检测和追踪，包含异常处理机制。
    """
    cap = None
    out = None

    try:
        # 1. 加载模型
        print("正在加载 YOLOv8 模型...")
        model = YOLO('yolov8n.pt')

        # 获取 'person' 类别 ID
        class_names = model.names
        if 'person' not in class_names.values():
            print("错误：模型中未找到 'person' 类别")
            return
        person_class_id = list(class_names.keys())[list(class_names.values()).index('person')]

        # 2. 打开视频
        if not os.path.exists(video_path):
            print(f"错误：找不到视频文件 '{video_path}'")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("错误：无法打开视频文件")
            return

        # 获取视频属性
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 如果 FPS 获取失败 (0.0)，默认设为 25.0 以免写入报错
        if fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 检查总帧数
        show_progress_percent = True
        if total_frames <= 0:
            print("警告：无法获取视频总帧数，将仅显示当前帧数。")
            show_progress_percent = False
        else:
            print(f"视频信息：{frame_width}x{frame_height}, FPS: {fps}, 总帧数: {total_frames}")

        # 3. 配置视频写入器
        # macOS 推荐 'avc1'，Windows 推荐 'mp4v' 或 'avc1'
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            print("错误：无法创建输出视频文件，请检查编码器或路径权限。")
            return

        print(f"开始处理... (按 'q' 键退出预览，或 Ctrl+C 停止)")

        frame_count = 0
        unique_track_ids = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频播放结束。")
                break

            frame_count += 1

            # 进度打印
            if frame_count % 10 == 0:
                if show_progress_percent:
                    percent = (frame_count / total_frames) * 100
                    print(f"进度: {frame_count}/{total_frames} ({percent:.1f}%)", end='\r')
                else:
                    print(f"正在处理第 {frame_count} 帧...", end='\r')

            # 核心推理
            results = model.track(
                source=frame,
                stream=True,
                classes=[person_class_id],
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
                conf=0.4,
                iou=0.5
            )

            for result in results:
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy()
                    for tid in track_ids:
                        unique_track_ids.add(int(tid))

                    # 绘制结果
                    annotated_frame = result.plot(line_width=2, font_size=12)

                    # 显示当前检测人数
                    cv2.putText(annotated_frame, f"People: {len(track_ids)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    annotated_frame = frame

                # 显示窗口
                cv2.imshow("Video Person Tracking", annotated_frame)

                # 保存帧
                out.write(annotated_frame)

                # 检测按键 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户按下 'q' 键，正在停止...")
                    break

        # 正常结束循环后的统计
        print("\n" + "-" * 30)
        print("处理完成！")
        print(f"输出文件已保存: {output_path}")
        print(f"检测到不同人物总数 (Unique IDs): {len(unique_track_ids)}")
        print("-" * 30)

    except KeyboardInterrupt:
        print("\n\n[系统中断] 检测到用户强制停止 (Ctrl+C)。")
        print("正在安全清理资源...")

    except Exception as e:
        print(f"\n[发生未知错误]: {e}")

    finally:
        # 【关键】无论是否出错，都要释放资源
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print("资源已释放，程序退出。")


if __name__ == '__main__':
    video_file = 0  # 确保这个文件存在
    if not os.path.exists(video_file):
        print(f"提示：未找到 '{video_file}'，请放入视频文件或修改路径。")
    else:
        track_people_in_video(video_file)