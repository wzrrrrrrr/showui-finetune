import cv2
import os
import json
import shutil

# --- 1. 配置 ---
# 你的原始图片所在的文件夹
SOURCE_IMAGE_DIR = "Boss_imgs"
# 你想生成的训练集文件夹名称
OUTPUT_DATASET_NAME = "my_dataset"
# --- 结束配置 ---

# 全局变量，用于存储鼠标事件的状态
drawing = False
start_point = (-1, -1)
current_box = None


def draw_rectangle(event, x, y, flags, param):
    """OpenCV的鼠标回调函数，用于处理鼠标事件"""
    global start_point, drawing, current_box, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        # 复制原始图像，以免在上面永久绘制多个预览框
        img_display = img.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 在图像副本上绘制一个临时的、移动的矩形作为预览
            img_temp = img_display.copy()
            cv2.rectangle(img_temp, start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Annotator", img_temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 确保 x_end > x_start, y_end > y_start
        x_start, y_start = start_point
        x_end, y_end = x, y

        box_start_x = min(x_start, x_end)
        box_start_y = min(y_start, y_end)
        box_end_x = max(x_start, x_end)
        box_end_y = max(y_start, y_end)

        current_box = (box_start_x, box_start_y, box_end_x, box_end_y)

        # 在主显示图像上绘制最终的、固定的矩形
        cv2.rectangle(img_display, (box_start_x, box_start_y), (box_end_x, box_end_y), (0, 0, 255), 2)
        cv2.imshow("Annotator", img_display)


def main():
    global img, img_display, current_box

    # --- 设置输出目录 ---
    output_dir = OUTPUT_DATASET_NAME
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)

    print(f"训练集将保存在: ./{output_dir}/")

    # 存储所有图片标注信息的列表
    final_dataset = []

    image_files = sorted([f for f in os.listdir(SOURCE_IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])

    for image_file in image_files:
        print("\n" + "=" * 50)
        print(f"正在处理图片: {image_file}")

        image_path = os.path.join(SOURCE_IMAGE_DIR, image_file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"  [警告] 无法读取图片: {image_path}, 已跳过。")
            continue

        # 复制图片到目标文件夹
        shutil.copy(image_path, os.path.join(output_images_dir, image_file))

        img_height, img_width, _ = img.shape
        img_display = img.copy()

        # 为当前图片创建标注数据结构
        current_image_data = {
            "img_url": image_file,
            "img_size": [img_width, img_height],
            "element": [],
            "element_size": 0
        }

        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", draw_rectangle)

        print("\n--- 操作指南 ---")
        print("1. 在图片上拖动鼠标左键来画一个框。")
        print("2. 画完框后，按 's' 键保存这个框。")
        print("3. 在控制台输入说明文字，然后选择类型 (text/image)。")
        print("4. 重复以上步骤为当前图片添加更多框。")
        print("5. 按 'n' 键完成当前图片，进入下一张。")
        print("6. 按 'q' 键随时退出并保存所有已完成的标注。")
        print("----------------\n")

        while True:
            cv2.imshow("Annotator", img_display)
            key = cv2.waitKey(1) & 0xFF

            # 按 's' 保存当前画的框
            if key == ord('s'):
                if current_box:
                    instruction = input(">> 请输入这个框的说明 (instruction): ")
                    if not instruction:
                        print("  [警告] 说明不能为空，此标注未保存。")
                        continue

                    # ######################### MODIFICATION START #########################
                    # 增加了选择类型的逻辑
                    data_type = ""
                    while data_type not in ["text", "image"]:
                        type_choice = input(">> 这是 'text' 还是 'image'? (输入 t 或 i): ").lower()
                        if type_choice == 't':
                            data_type = "text"
                        elif type_choice == 'i':
                            data_type = "image"
                        else:
                            print("  [错误] 无效输入，请输入 't' (代表text) 或 'i' (代表image)。")
                    # ########################## MODIFICATION END ##########################

                    # 转换像素坐标为归一化坐标
                    px_min, py_min, px_max, py_max = current_box
                    norm_bbox = [
                        px_min / img_width,
                        py_min / img_height,
                        px_max / img_width,
                        py_max / img_height
                    ]
                    norm_point = [
                        (norm_bbox[0] + norm_bbox[2]) / 2,
                        (norm_bbox[1] + norm_bbox[3]) / 2
                    ]

                    # 添加到当前图片的元素列表
                    element_data = {
                        "instruction": instruction,
                        "bbox": norm_bbox,
                        "data_type": data_type,  # 使用用户选择的值
                        "point": norm_point
                    }
                    current_image_data["element"].append(element_data)
                    current_image_data["element_size"] += 1

                    # 在图像上用绿色永久标记已保存的框和文字
                    cv2.rectangle(img, (px_min, py_min), (px_max, py_max), (0, 255, 0), 2)
                    cv2.putText(img, f"#{current_image_data['element_size']}:{data_type}", (px_min, py_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    img_display = img.copy()

                    print(f"  [成功] 框 #{current_image_data['element_size']} ({data_type}) 已保存: '{instruction}'")
                    current_box = None
                else:
                    print("  [提示] 请先画一个框再按 's'。")

            # 按 'n' 进入下一张图片
            elif key == ord('n'):
                if current_image_data["element_size"] > 0:
                    final_dataset.append(current_image_data)
                    print(f"\n图片 {image_file} 完成，共 {current_image_data['element_size']} 个标注。")
                else:
                    print(f"\n图片 {image_file} 已跳过（无标注）。")
                break  # 退出当前图片的循环

            # 按 'q' 退出整个程序
            elif key == ord('q'):
                # 检查当前图片是否有未保存的标注
                if current_image_data["element_size"] > 0:
                    final_dataset.append(current_image_data)
                    print(f"\n图片 {image_file} 完成，共 {current_image_data['element_size']} 个标注。")

                # 保存所有数据到JSON文件
                output_json_path = os.path.join(OUTPUT_DATASET_NAME, "metadata.json")
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(final_dataset, f, indent=4, ensure_ascii=False)

                print("\n" + "=" * 50)
                print("程序退出。")
                print(f"所有已完成的标注已保存到: {output_json_path}")
                cv2.destroyAllWindows()
                return

    # 所有图片处理完毕后
    output_json_path = os.path.join(OUTPUT_DATASET_NAME, "metadata.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("所有图片都已处理完毕！")
    print(f"训练集已生成在: {output_json_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()