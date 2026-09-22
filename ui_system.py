import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk


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
# إعدادات التطبيق — تصميم داكن مع توهج ذهبي
# ============================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# الألوان المستوحاة من صورة القلب
BG_COLOR = "#000000"          # أسود عميق
CARD_COLOR = "#0a0a0a"        # أسود للبطاقات
GLOW_GOLD = "#ffaa00"         # ذهبي متوهج
GLOW_ORANGE = "#ff8800"       # برتقالي متوهج
TEXT_WHITE = "#f0f0f0"        # أبيض ناعم
TEXT_DIM = "#888888"          # رمادي للنصوص الثانوية
SUCCESS_GREEN = "#66ff99"     # أخضر للنجاح
ERROR_RED = "#ff4444"         # أحمر للأخطاء

app = ctk.CTk()
app.geometry("700x850")
app.title("نظام ذكي للتنبؤ بأمراض القلب")
app.configure(fg_color=BG_COLOR)

ctk.FontManager.load_font("Tajawal.ttf")


# ============================================
# العنوان الرئيسي
# ============================================
title_label = ctk.CTkLabel(
    app,
    text="🫀 نظام ذكي للتنبؤ بأمراض القلب",
    font=("Tajawal", 26, "bold"),
    text_color=GLOW_GOLD
)
title_label.pack(pady=(25, 5))

subtitle_label = ctk.CTkLabel(
    app,
    text="Intelligent Heart Disease Prediction System",
    font=("Tajawal", 12),
    text_color=TEXT_DIM
)
subtitle_label.pack(pady=(0, 15))


# ============================================
# 🎬 عرض الفيديو heart_icon.mp4 (باستخدام OpenCV + ImageTk)
# ============================================
video_frame = ctk.CTkFrame(
    app,
    fg_color=CARD_COLOR,
    corner_radius=20,
    border_width=2,
    border_color=GLOW_ORANGE,
    width=280,
    height=280
)
video_frame.pack(pady=10)
video_frame.pack_propagate(False)

video_label = ctk.CTkLabel(video_frame, text="", width=260, height=260)
video_label.pack(expand=True)

# متغيرات الفيديو
video_cap = None
video_running = False
video_paused = False


def start_video():
    global video_cap, video_running
    video_path = "heart_icon.mp4"
    if not os.path.exists(video_path):
        print("⚠️ heart_icon.mp4 غير موجود")
        return
    video_cap = cv2.VideoCapture(video_path)
    video_running = True
    play_video_frame()


def play_video_frame():
    global video_cap, video_running, video_paused

    if not video_running or video_cap is None:
        return

    if video_paused:
        video_label.after(50, play_video_frame)
        return

    ret, frame = video_cap.read()
    if not ret:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = video_cap.read()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((260, 260), Image.LANCZOS)

        # استخدام ImageTk.PhotoImage (أكثر استقراراً من CTkImage)
        photo = ImageTk.PhotoImage(img)
        video_label.configure(image=photo, text="")
        video_label.image = photo

    video_label.after(50, play_video_frame)  # 20 FPS


def stop_video():
    global video_cap, video_running
    video_running = False
    if video_cap is not None:
        video_cap.release()
        video_cap = None


# ============================================
# صورة ECG المحمّلة
# ============================================
image_frame = ctk.CTkFrame(
    app,
    fg_color=CARD_COLOR,
    corner_radius=15,
    border_width=1,
    border_color="#333333",
    width=220,
    height=220
)
image_frame.pack(pady=10)

image_label = ctk.CTkLabel(image_frame, text="", width=200, height=200)
image_label.pack(expand=True)

# في البداية مخفي
image_frame.pack_forget()


# ============================================
# النصوص
# ============================================
status_label = ctk.CTkLabel(app, text="", font=("Tajawal", 14), text_color=SUCCESS_GREEN)
status_label.pack(pady=5)

result_label = ctk.CTkLabel(app, text="", font=("Tajawal", 18, "bold"), text_color=GLOW_GOLD)
result_label.pack(pady=5)

arabic_result_label = ctk.CTkLabel(app, text="", font=("Tajawal", 20, "bold"), text_color=GLOW_ORANGE)
arabic_result_label.pack(pady=5)


# ============================================
# شريط التقدم
# ============================================
progress_bar = ctk.CTkProgressBar(
    app,
    mode="determinate",
    width=400,
    height=10,
    progress_color=GLOW_GOLD,
    fg_color="#1a1a1a"
)
progress_bar.pack(pady=10)
progress_bar.pack_forget()


# ============================================
# الوظائف
# ============================================
def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    img = Image.fromarray(image)
    img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))

    image_label.configure(image=img_ctk, text="")
    image_label.image = img_ctk
    image_label.file_path = file_path

    image_frame.pack(pady=10)
    status_label.configure(text="✅ تم تحميل الصورة بنجاح", text_color=SUCCESS_GREEN)


def predict_image():
    global video_paused
    if not hasattr(image_label, 'file_path'):
        result_label.configure(text="⚠️ يرجى تحميل صورة أولاً", text_color=ERROR_RED)
        return

    video_paused = True
    progress_bar.pack(pady=10)
    progress_bar.set(0)

    image = cv2.imread(image_label.file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    def complete_prediction():
        global video_paused
        prediction = model.predict(image, verbose=0)
        predicted_class = np.argmax(prediction)
        english_result, arabic_result = class_labels[predicted_class]

        result_label.configure(text=f"🔍 Result: {english_result}", text_color=GLOW_GOLD)
        arabic_result_label.configure(text=f"💙 النتيجة: {arabic_result}", text_color=GLOW_ORANGE)

        progress_bar.pack_forget()
        video_paused = False

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
    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                              filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"{result_label.cget('text')}\n{arabic_result_label.cget('text')}")


def reset_ui():
    image_label.configure(image="", text="")
    if hasattr(image_label, 'file_path'):
        del image_label.file_path
    image_frame.pack_forget()
    status_label.configure(text="")
    result_label.configure(text="")
    arabic_result_label.configure(text="")
    progress_bar.pack_forget()


# ============================================
# الأزرار — تصميم دائري مع توهج
# ============================================
button_frame = ctk.CTkFrame(app, fg_color="transparent")
button_frame.pack(pady=15)

# زر تحميل صورة
load_button = ctk.CTkButton(
    button_frame, text="📂 تحميل صورة",
    font=("Tajawal", 14, "bold"),
    command=load_image,
    fg_color=GLOW_ORANGE,
    hover_color=GLOW_GOLD,
    text_color="#000000",
    corner_radius=25,
    width=130, height=45,
    border_width=0
)
load_button.pack(side="left", padx=6)

# زر التنبؤ
predict_button = ctk.CTkButton(
    button_frame, text="🔍 التنبؤ",
    font=("Tajawal", 14, "bold"),
    command=predict_image,
    fg_color=GLOW_GOLD,
    hover_color=GLOW_ORANGE,
    text_color="#000000",
    corner_radius=25,
    width=130, height=45
)
predict_button.pack(side="left", padx=6)

# زر حفظ
save_button = ctk.CTkButton(
    button_frame, text="💾 حفظ",
    font=("Tajawal", 14, "bold"),
    command=save_result,
    fg_color="#1a1a1a",
    hover_color="#2a2a2a",
    text_color=GLOW_GOLD,
    corner_radius=25,
    width=110, height=45,
    border_width=2,
    border_color=GLOW_GOLD
)
save_button.pack(side="left", padx=6)

# زر إعادة ضبط
reset_button = ctk.CTkButton(
    button_frame, text="🔄 إعادة ضبط",
    font=("Tajawal", 14, "bold"),
    command=reset_ui,
    fg_color="#1a1a1a",
    hover_color="#2a2a2a",
    text_color=GLOW_ORANGE,
    corner_radius=25,
    width=130, height=45,
    border_width=2,
    border_color=GLOW_ORANGE
)
reset_button.pack(side="left", padx=6)


# ============================================
# زر الخروج
# ============================================
exit_button = ctk.CTkButton(
    app, text="❌ خروج",
    font=("Tajawal", 14, "bold"),
    command=lambda: (stop_video(), app.quit()),
    fg_color="transparent",
    hover_color="#330000",
    text_color=ERROR_RED,
    corner_radius=20,
    width=150, height=40,
    border_width=2,
    border_color=ERROR_RED
)
exit_button.pack(pady=10)


# ============================================
# تشغيل الفيديو
# ============================================
app.after(100, start_video)

app.mainloop()