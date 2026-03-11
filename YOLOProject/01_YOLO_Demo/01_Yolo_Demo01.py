from ultralytics import YOLO
import cv2


def main():
    # 1. 加载模型
    # 自动下载最新的 yolov8n.pt (nano版本，速度最快，适合入门)
    # 可选模型: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    print("正在加载模型...")
    model = YOLO('yolov8n.pt')

    # ---------------------------------------------------------
    # 场景 A: 单张图片检测
    # ---------------------------------------------------------
    print("\n[场景 A] 开始进行图片检测...")
    # 替换为你本地的图片路径，或者使用网络URL
    source_img = 'https://ultralytics.com/images/bus.jpg'

    # 执行预测
    # save=True: 保存结果图到 runs/detect/predict
    # show=False: 在服务器上运行时建议关闭，本地运行可开启弹窗
    results = model.predict(source=source_img, save=True, show=True)

    # 打印检测结果
    result = results[0]
    print(f"检测到 {len(result.boxes)} 个目标")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls_id]
        print(f"  - 类别: {name}, 置信度: {conf:.2f}")

    # ---------------------------------------------------------
    # 场景 B: 实时摄像头检测 (按 'q' 键退出)
    # ---------------------------------------------------------
    print("\n[场景 B] 启动摄像头检测 (按 'q' 退出)...")

    # 打开默认摄像头 (0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 执行推理
        # stream=True 对于视频流/摄像头非常重要，它可以生成器模式返回结果，节省内存
        results = model.predict(source=frame, stream=True, verbose=False)

        # 获取当前帧的结果
        result = results[0]

        # 绘制结果 (YOLOv8自带绘图功能，也可以手动用cv2画)
        annotated_frame = result.plot()

        # 显示画面
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Demo结束")


if __name__ == '__main__':
    main()
