import os
import cv2
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk


# ============================================
# إعدادات التطبيق
# ============================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("900x750")
app.title("نظام ذكي للتنبؤ بأمراض القلب")
app.configure(fg_color="#0a0a15")

try:
    ctk.FontManager.load_font("Tajawal.ttf")
    FONT = "Tajawal"
except:
    FONT = "Arial"


# ============================================
# تحميل النموذج
# ============================================
model = keras.models.load_model('cnn_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class_labels = {
    0: ("Myocardial Infarction", "احتشاء عضلة القلب"),
    1: ("Fusion Beats", "نبضات قلب مدمجة"),
    2: ("Supraventricular Ectopic Beats", "نبضات فوق البطينية الهاجرة"),
    3: ("Normal ECG", "تخطيط قلب طبيعي"),
    4: ("Unknown Beats", "نبضات غير معروفة"),
    5: ("Ventricular Ectopic Beats", "نبضات بطينية هاجرة")
}


# ============================================
# الخلفية + Canvas
# ============================================
background_frame = ctk.CTkFrame(app, fg_color="#0a0a15", corner_radius=0)
background_frame.place(relwidth=1, relheight=1)

canvas = ctk.CTkCanvas(
    background_frame,
    width=900,
    height=750,
    bg="#0a0a15",
    highlightthickness=0
)
canvas.pack(fill="both", expand=True)


# === شبكة الخلفية ===
def draw_grid():
    grid_color = "#1a1a2e"
    step = 30
    for x in range(0, 900, step):
        canvas.create_line(x, 0, x, 750, fill=grid_color, width=1)
    for y in range(0, 750, step):
        canvas.create_line(0, y, 900, y, fill=grid_color, width=1)

draw_grid()


# ============================================
# 💓 القلب النابض
# ============================================
heart_image_original = None
canvas_heart_id = None
heart_scale_state = [1.0]       # حجم القلب الحالي
heart_scale_direction = [-0.005]  # الاتجاه (يتمدد أم ينكمش)
heart_glow_state = [0]           # لتتبع حالة التوهج

# تحميل صورة القلب الأصلية
if os.path.exists("heart_icon.png"):
    heart_image_original = Image.open("heart_icon.png").convert("RGBA")


def update_heart_beat():
    """تحديث نبضات القلب — يتوسع وينكمش"""
    global canvas_heart_id, heart_image_original
    
    if heart_image_original is None:
        return
    
    # تحديث الحجم
    heart_scale_state[0] += heart_scale_direction[0]
    
    if heart_scale_state[0] >= 1.08:
        heart_scale_state[0] = 1.08
        heart_scale_direction[0] = -0.005
    elif heart_scale_state[0] <= 0.95:
        heart_scale_state[0] = 0.95
        heart_scale_direction[0] = 0.005
    
    # إعادة تحجيم القلب
    size = int(320 * heart_scale_state[0])
    heart_resized = heart_image_original.resize((size, size), Image.LANCZOS)
    heart_photo = ImageTk.PhotoImage(heart_resized)
    
    # حفظ مرجع (لمنع جمع القمامة)
    canvas.heart_photo_ref = heart_photo
    
    # حذف القلب القديم ورسم الجديد
    if canvas_heart_id:
        canvas.delete(canvas_heart_id)
    
    canvas_heart_id = canvas.create_image(
        450, 400,
        image=heart_photo,
        anchor="center"
    )
    
    # جدولة التحديث القادم (50ms = 20 FPS)
    app.after(50, update_heart_beat)


# بدء نبضات القلب
if heart_image_original is not None:
    update_heart_beat()


# ============================================
# 📈 خط ECG المتحرك
# ============================================
ecg_offset = [0]  # مقدار الإزاحة
ecg_lines = []     # مراجع الخطوط لمسحها

# موجة نبضة القلب (نمط)
ECG_PATTERN = [
    0, 0, 0, 0, 0, 0,
    2, 5, 8, 5, 2, 0,
    0, 0, 0, 0, 0, 0,
    -3, -8, -12, -8, -3, 0,
    0, 0, 0, 0, 0, 0,
    0, 0,
    40, 80, 60, 20, 0,        # R peak (قمة)
    -20, -40, -20, 0,
    0, 0, 0, 0, 0, 0,
    3, 6, 3, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0
]

ECG_PATTERN_LEN = len(ECG_PATTERN)
BASE_Y = 430           # الخط الأساسي لـ ECG
POINT_SPACING = 4      # المسافة بين النقاط


def draw_ecg_animated():
    """رسم خط ECG متحرك مع توهج"""
    global ecg_lines
    
    # مسح الخطوط القديمة
    for line in ecg_lines:
        canvas.delete(line)
    ecg_lines = []
    
    # حساب النقاط
    ecg_points = []
    for x in range(0, 920, POINT_SPACING):
        # حساب الموضع في النمط (مع الإزاحة للحركة)
        pattern_index = (x // POINT_SPACING + ecg_offset[0]) % ECG_PATTERN_LEN
        val = ECG_PATTERN[pattern_index]
        y = BASE_Y - val
        ecg_points.append((x, y))
    
    # رسم طبقات التوهج (خلف الخط الأساسي)
    glow_layers = [
        (14, "#1a1000"),   # طبقة خارجية داكنة
        (10, "#3a2500"),   # طبقة وسطى
        (7, "#5a3a00"),    # طبقة داخلية
        (4, "#7a5000"),    # قريبة من الأساسي
    ]
    
    for width, color in glow_layers:
        for i in range(len(ecg_points) - 1):
            line_id = canvas.create_line(
                ecg_points[i][0], ecg_points[i][1],
                ecg_points[i + 1][0], ecg_points[i + 1][1],
                fill=color, width=width, smooth=True
            )
            ecg_lines.append(line_id)
    
    # الخط الأساسي الأصفر الساطع
    for i in range(len(ecg_points) - 1):
        line_id = canvas.create_line(
            ecg_points[i][0], ecg_points[i][1],
            ecg_points[i + 1][0], ecg_points[i + 1][1],
            fill="#ffdd00", width=2, smooth=True
        )
        ecg_lines.append(line_id)
    
    # إزاحة الخطوة التالية (لإنشاء حركة)
    ecg_offset[0] = (ecg_offset[0] + 1) % ECG_PATTERN_LEN
    
    # جدولة التحديث القادم (30ms = 33 FPS)
    app.after(30, draw_ecg_animated)


# بدء خط ECG المتحرك
draw_ecg_animated()


# ============================================
# العنوان
# ============================================
title_bg = ctk.CTkFrame(
    background_frame,
    fg_color="#1a1a2e",
    corner_radius=12,
    border_width=2,
    border_color="#4a9eff"
)
title_bg.place(relx=0.5, y=40, anchor="center")

title_label = ctk.CTkLabel(
    title_bg,
    text="🫀 نظام ذكي للتنبؤ بأمراض القلب",
    font=(FONT, 22, "bold"),
    text_color="#ffffff"
)
title_label.pack(padx=30, pady=12)


# ============================================
# منطقة عرض صورة ECG المحمّلة
# ============================================
image_display_label = ctk.CTkLabel(
    background_frame,
    text="",
    width=250,
    height=250,
    fg_color="transparent"
)
image_display_label.place(relx=0.5, y=300, anchor="center")
image_display_label.place_forget()


# ============================================
# النصوص
# ============================================
status_label = ctk.CTkLabel(
    background_frame,
    text="",
    font=(FONT, 14, "bold"),
    text_color="#66ff99"
)
status_label.place(relx=0.5, y=580, anchor="center")

result_label = ctk.CTkLabel(
    background_frame,
    text="",
    font=(FONT, 18, "bold"),
    text_color="#4a9eff"
)
result_label.place(relx=0.5, y=615, anchor="center")

arabic_result_label = ctk.CTkLabel(
    background_frame,
    text="",
    font=(FONT, 20, "bold"),
    text_color="#ffdd00"
)
arabic_result_label.place(relx=0.5, y=650, anchor="center")


# ============================================
# شريط التقدم
# ============================================
progress_bar = ctk.CTkProgressBar(
    background_frame,
    mode="determinate",
    width=400,
    height=8,
    progress_color="#4a9eff",
    fg_color="#1a1a2e"
)
progress_bar.place(relx=0.5, y=685, anchor="center")
progress_bar.place_forget()


# ============================================
# الوظائف
# ============================================
def load_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if not file_path:
        return
    
    # إخفاء القلب عند تحميل صورة جديدة
    if canvas_heart_id:
        canvas.itemconfigure(canvas_heart_id, state="hidden")
    
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (250, 250))
    
    img = Image.fromarray(image)
    img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(250, 250))
    
    image_display_label.configure(image=img_ctk, text="")
    image_display_label.image = img_ctk
    image_display_label.file_path = file_path
    image_display_label.place(relx=0.5, y=300, anchor="center")
    
    status_label.configure(text="✅ تم تحميل الصورة بنجاح", text_color="#66ff99")


def predict_image():
    if not hasattr(image_display_label, 'file_path'):
        result_label.configure(text="⚠️ يرجى تحميل صورة أولاً", text_color="#ff4444")
        return
    
    progress_bar.place(relx=0.5, y=685, anchor="center")
    progress_bar.set(0)
    
    image = cv2.imread(image_display_label.file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    
    def complete_prediction():
        prediction = model.predict(image, verbose=0)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        
        english_result, arabic_result = class_labels[predicted_class]
        
        result_label.configure(
            text=f"🔍 Result: {english_result}",
            text_color="#4a9eff"
        )
        arabic_result_label.configure(
            text=f"💙 النتيجة: {arabic_result}   ({confidence:.1f}%)",
            text_color="#ffdd00"
        )
        
        progress_bar.place_forget()
    
    def update_progress(value):
        progress_bar.set(value)
        if value < 1.0:
            app.after(80, lambda: update_progress(value + 0.1))
        else:
            complete_prediction()
    
    update_progress(0.1)


def save_result():
    if result_label.cget("text") == "":
        return
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt")]
    )
    
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"{result_label.cget('text')}\n")
            f.write(f"{arabic_result_label.cget('text')}\n")


def reset_ui():
    image_display_label.configure(image="", text="")
    if hasattr(image_display_label, 'file_path'):
        del image_display_label.file_path
    image_display_label.place_forget()
    
    if canvas_heart_id:
        canvas.itemconfigure(canvas_heart_id, state="normal")
    
    status_label.configure(text="")
    result_label.configure(text="")
    arabic_result_label.configure(text="")
    progress_bar.place_forget()


# ============================================
# شريط الأزرار
# ============================================
button_bar = ctk.CTkFrame(
    background_frame,
    fg_color="#1a1a2e",
    corner_radius=15,
    border_width=2,
    border_color="#4a9eff",
    height=60
)
button_bar.place(relx=0.5, y=710, anchor="center")

load_btn = ctk.CTkButton(
    button_bar,
    text="📂 تحميل صورة",
    font=(FONT, 14, "bold"),
    command=load_image,
    fg_color="#4a9eff",
    hover_color="#3a8eef",
    text_color="#ffffff",
    corner_radius=10,
    width=140,
    height=42
)
load_btn.pack(side="left", padx=8, pady=9)

predict_btn = ctk.CTkButton(
    button_bar,
    text="🔍 التنبؤ بالنتيجة",
    font=(FONT, 14, "bold"),
    command=predict_image,
    fg_color="#4a9eff",
    hover_color="#3a8eef",
    text_color="#ffffff",
    corner_radius=10,
    width=150,
    height=42
)
predict_btn.pack(side="left", padx=8, pady=9)

save_btn = ctk.CTkButton(
    button_bar,
    text="💾 حفظ النتيجة",
    font=(FONT, 14, "bold"),
    command=save_result,
    fg_color="#4a9eff",
    hover_color="#3a8eef",
    text_color="#ffffff",
    corner_radius=10,
    width=140,
    height=42
)
save_btn.pack(side="left", padx=8, pady=9)

reset_btn = ctk.CTkButton(
    button_bar,
    text="🔄 إعادة ضبط",
    font=(FONT, 14, "bold"),
    command=reset_ui,
    fg_color="#4a9eff",
    hover_color="#3a8eef",
    text_color="#ffffff",
    corner_radius=10,
    width=130,
    height=42
)
reset_btn.pack(side="left", padx=8, pady=9)

exit_btn = ctk.CTkButton(
    button_bar,
    text="❌ خروج",
    font=(FONT, 14, "bold"),
    command=app.quit,
    fg_color="#c41e3a",
    hover_color="#a01830",
    text_color="#ffffff",
    corner_radius=10,
    width=100,
    height=42
)
exit_btn.pack(side="left", padx=8, pady=9)


# ============================================
# تشغيل التطبيق
# ============================================
app.mainloop()