# 终极版 v5 (AI去背效果优化)
import os
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from PIL import Image, ImageOps
import cv2
import numpy as np
import urllib.request
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# --- 全局配置 (无变化) ---
ID_PHOTO_SPECS = {
    "1寸 (2.5x3.5cm)": {"size_px": (295, 413), "dpi": (300, 300), "suffix": "_1inch"},
    "小二寸 (3.3x4.8cm)": {"size_px": (390, 567), "dpi": (300, 300), "suffix": "_small2inch"},
    "二寸 (3.5x5.3cm)": {"size_px": (413, 626), "dpi": (300, 300), "suffix": "_2inch"},
}
CASCADE_FILE = "haarcascade_frontalface_default.xml"
CASCADE_URL = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{CASCADE_FILE}"

# --- 辅助函数 (无变化) ---
def ensure_cascade_file():
    if os.path.exists(CASCADE_FILE): return True
    print(f"AI模型文件 '{CASCADE_FILE}' 不存在，正在尝试自动下载...")
    try:
        with urllib.request.urlopen(CASCADE_URL) as response, open(CASCADE_FILE, 'wb') as out_file:
            out_file.write(data := response.read())
        print("模型下载成功！")
        return True
    except Exception as e:
        messagebox.showerror("下载失败", f"无法下载人脸识别模型: {e}\n请检查网络连接。")
        return False

# --- 核心处理逻辑 (AI去背优化) ---
def process_single_id_photo_with_ai(params, app_instance):
    source_file_path = params['file_path']
    if not source_file_path:
        messagebox.showwarning("警告", "请先选择一个图片文件！")
        return

    print("\n--- 开始新任务 (v5 - AI去背优化版) ---")
    print(f"处理文件: {source_file_path}")

    app_instance.root.config(cursor="watch")
    app_instance.root.update_idletasks()

    try:
        if params['remove_bg'] and not REMBG_AVAILABLE:
            messagebox.showerror("错误", "rembg库未安装，无法使用去背功能。\n请在命令行运行: pip install rembg")
            return
        if not ensure_cascade_file(): return
        
        face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
        
        spec = ID_PHOTO_SPECS.get(params['id_spec_key'])
        target_size_px = spec["size_px"]
        output_suffix = spec["suffix"]

        img = Image.open(source_file_path)
        
        if params['remove_bg']:
            print("步骤 1: 正在进行AI智能去背 (高级模式)...")
            try:
                # ========== 算法升级点 ==========
                # 1. 启用 Alpha Matting 精细处理边缘
                # 2. 调用更专业的人像分割模型
                foreground = remove(
                    img,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                    model='u2net_human_seg' # 注意：首次使用此模型会自动下载
                )
                # ===============================

                background = Image.new("RGB", foreground.size, params['bg_color'])
                background.paste(foreground, (0, 0), foreground)
                img = background
                output_suffix += f"_{params['bg_name']}_bg"
                print("背景更换完成。")
            except Exception as e:
                print(f"AI去背失败: {e}")
                messagebox.showerror("AI去背失败", f"无法去除背景，请检查rembg库或图片文件。\n可能是新模型下载失败，请检查网络后重试。\n错误: {e}")
                return
        
        img = img.convert("RGB")
        
        # 步骤 2 和 3 (智能构图、保存) 与上一版完全相同，此处省略以保持清晰
        # ... (此处省略了完整的构图和保存代码，它们与上一版健壮版完全相同) ...
        print("步骤 2: 正在进行人脸检测与智能构图...")
        cv_img = np.array(img)
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        
        cropped_img = None
        if len(faces) > 0:
            print(f"检测到 {len(faces)} 张人脸，将使用最大的一张进行构图。")
            main_face = max(faces, key=lambda rect: rect[2] * rect[3])
            
            x, y, w, h = main_face
            HAIR_FACTOR, HEAD_HEIGHT_RATIO, HEAD_TOP_MARGIN_RATIO = 0.25, 0.7, 0.1
            estimated_head_top_y = y - h * HAIR_FACTOR
            estimated_head_height = h * (1 + HAIR_FACTOR)
            crop_area_height = estimated_head_height / (1 - HEAD_TOP_MARGIN_RATIO - (1 - HEAD_HEIGHT_RATIO))
            target_aspect_ratio = target_size_px[0] / target_size_px[1]
            crop_area_width = crop_area_height * target_aspect_ratio
            crop_y1 = estimated_head_top_y - crop_area_height * HEAD_TOP_MARGIN_RATIO
            face_center_x = x + w / 2
            crop_x1 = face_center_x - crop_area_width / 2
            crop_x1, crop_y1 = max(0, crop_x1), max(0, crop_y1)
            crop_x2, crop_y2 = crop_x1 + crop_area_width, crop_y1 + crop_area_height
            if crop_x2 > img.width: crop_x2 = img.width
            if crop_y2 > img.height: crop_y2 = img.height
            crop_box = (int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2))
            wide_crop_img = img.crop(crop_box)
            cropped_img = wide_crop_img.resize(target_size_px, Image.Resampling.LANCZOS)
            print("智能构图完成。")
        else:
            print("未检测到人脸，将使用传统中心裁剪。")
            messagebox.showwarning("AI未识别", "未在此图片中检测到人脸，将使用传统中心裁剪。")
            cropped_img = ImageOps.fit(img, target_size_px, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

        print("步骤 3: 正在添加白边并保存文件...")
        final_img = cropped_img
        if params['add_border']:
            border_width_px = 10
            background = Image.new('RGB', (target_size_px[0] + 2 * border_width_px, target_size_px[1] + 2 * border_width_px), (255, 255, 255))
            background.paste(cropped_img, (border_width_px, border_width_px))
            final_img = background

        dir_name, base_name = os.path.split(source_file_path)
        file_name, file_ext = os.path.splitext(base_name)
        output_file_path = os.path.join(dir_name, f"{file_name}{output_suffix}.jpg")

        final_img.save(output_file_path, "JPEG", quality=95, dpi=spec["dpi"])
        print(f"任务成功完成！文件已保存至: {output_file_path}")
        messagebox.showinfo("成功", f"证件照已生成！\n\n已保存至:\n{output_file_path}")

    except Exception as e:
        print(f"!!! 任务失败，发生未知错误: {e}")
        messagebox.showerror("处理失败", f"发生未知错误，请查看命令行窗口获取详细信息。\n错误: {e}")
    finally:
        app_instance.root.config(cursor="")

# --- 图形用户界面 (GUI - 无变化) ---
# GUI代码与上一版完全相同，此处省略。
# 请直接使用上一版的App类代码，它已经可以驱动新的处理逻辑。
class App:
    def __init__(self, root):
        self.root = root
        self.file_path = tk.StringVar()
        self.id_spec_var = tk.StringVar(value=list(ID_PHOTO_SPECS.keys())[0])
        self.add_border_var = tk.BooleanVar(value=True)
        self.remove_bg_var = tk.BooleanVar(value=False)
        self.bg_color_choice = tk.StringVar(value="blue") 
        self.custom_color_rgb = (255, 255, 255)
        self.custom_color_hex = "#ffffff"
        root.title("AI全能证件照生成器 (v5-效果增强)")
        root.geometry("520x520")
        root.resizable(False, False)
        main_frame = tk.Frame(root)
        main_frame.pack(padx=20, pady=15, fill="both", expand=True)
        file_frame = tk.LabelFrame(main_frame, text="1. 选择单个图片文件", font=("Helvetica", 12, "bold"))
        file_frame.pack(fill="x", pady=5)
        tk.Entry(file_frame, textvariable=self.file_path, width=55, state='readonly').pack(side="left", padx=10, pady=10, fill="x", expand=True)
        tk.Button(file_frame, text="浏览...", command=self.select_file).pack(side="right", padx=10, pady=10)
        spec_frame = tk.LabelFrame(main_frame, text="2. 选择证件照尺寸", font=("Helvetica", 12, "bold"))
        spec_frame.pack(fill="x", pady=10)
        for spec_name in ID_PHOTO_SPECS.keys():
            tk.Radiobutton(spec_frame, text=spec_name, variable=self.id_spec_var, value=spec_name, font=("Helvetica", 10)).pack(anchor="w", padx=20, pady=2)
        bg_frame = tk.LabelFrame(main_frame, text="3. 背景处理 (可选)", font=("Helvetica", 12, "bold"))
        bg_frame.pack(fill="x", pady=10, ipady=5)
        tk.Checkbutton(bg_frame, text="去除背景并更换颜色 (效果增强)", variable=self.remove_bg_var, font=("Helvetica", 10, 'bold'), command=self.check_rembg).pack(anchor="w", padx=15)
        color_options_frame = tk.Frame(bg_frame)
        color_options_frame.pack(pady=5)
        tk.Radiobutton(color_options_frame, text="蓝底", variable=self.bg_color_choice, value="blue").pack(side="left", padx=5)
        tk.Radiobutton(color_options_frame, text="红底", variable=self.bg_color_choice, value="red").pack(side="left", padx=5)
        tk.Radiobutton(color_options_frame, text="白底", variable=self.bg_color_choice, value="white").pack(side="left", padx=5)
        tk.Radiobutton(color_options_frame, text="自定义", variable=self.bg_color_choice, value="custom").pack(side="left", padx=5)
        self.custom_color_swatch = tk.Label(color_options_frame, text="  ", bg=self.custom_color_hex, relief="sunken", borderwidth=1)
        self.custom_color_swatch.pack(side="left", padx=5)
        self.custom_color_swatch.bind("<Button-1>", self.choose_custom_color)
        other_frame = tk.LabelFrame(main_frame, text="4. 其他选项", font=("Helvetica", 12, "bold"))
        other_frame.pack(fill="x", pady=10)
        tk.Checkbutton(other_frame, text="为照片添加白色边框", variable=self.add_border_var, font=("Helvetica", 10)).pack(anchor="w", padx=15, pady=5)
        tk.Button(main_frame, text="一键智能生成", command=self.start_processing, bg="#28a745", fg="white", font=("Helvetica", 14, "bold"), width=20, height=2).pack(pady=15)
    def check_rembg(self):
        if self.remove_bg_var.get() and not REMBG_AVAILABLE:
            self.remove_bg_var.set(False)
            messagebox.showerror("错误", "rembg库未安装，无法使用去背功能。\n请在命令行运行: pip install rembg")
    def choose_custom_color(self, event):
        color_code = colorchooser.askcolor(title="选择自定义背景色", initialcolor=self.custom_color_hex)
        if color_code and color_code[1]:
            self.custom_color_rgb = tuple(map(int, color_code[0]))
            self.custom_color_hex = color_code[1]
            self.custom_color_swatch.config(bg=self.custom_color_hex)
            self.bg_color_choice.set("custom")
    def select_file(self):
        file_types = [("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        file_path_selected = filedialog.askopenfilename(title="请选择一张图片", filetypes=file_types)
        if file_path_selected:
            self.file_path.set(file_path_selected)
    def start_processing(self):
        color_map = {
            "blue": ((67, 142, 219), "blue"), "red": ((255, 0, 0), "red"),
            "white": ((255, 255, 255), "white"), "custom": (self.custom_color_rgb, "custom"),
        }
        bg_color_tuple, bg_name = color_map.get(self.bg_color_choice.get())
        params = {
            'file_path': self.file_path.get(), 'id_spec_key': self.id_spec_var.get(),
            'add_border': self.add_border_var.get(), 'remove_bg': self.remove_bg_var.get(),
            'bg_color': bg_color_tuple, 'bg_name': bg_name
        }
        process_single_id_photo_with_ai(params, self)

if __name__ == "__main__":
    main_root = tk.Tk()
    app = App(main_root)
    main_root.mainloop()