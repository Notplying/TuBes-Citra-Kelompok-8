import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from tkinter import ttk
import cv2
from PIL import Image, ImageTk, ImageOps
from ultralytics import YOLO
import numpy as np

# Initialize YOLOv8 model
model = YOLO("best.pt")

# Warna
CLASS_COLORS = {
    "Car": (139, 0, 0),
    "Vacant": (147, 112, 219)
}

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Object Detection")
        self.root.configure(bg="#2e2e2e")
        
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12), padding=10, background="#6A0DAD", foreground="#4B0082")
        style.configure("TLabel", background="#2e2e2e", foreground="white", font=("Helvetica", 12))
        
        self.upload_button = ttk.Button(root, text="Upload Image", command=self.upload_image, style="TButton")
        self.upload_button.pack(pady=20)
        
        self.label_frame = ttk.Frame(root)
        self.label_frame.pack(pady=10)
        
        self.label = ttk.Label(self.label_frame)
        self.label.pack()
        
        self.inferred_label_frame = ttk.Frame(root)
        self.inferred_label_frame.pack(pady=10)
        
        self.inferred_label = ttk.Label(self.inferred_label_frame)
        self.inferred_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)
            self.infer_image(file_path)

    def display_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((400, 400), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(image)
        self.label.configure(image=img)
        self.label.image = img

    def infer_image(self, file_path):
        img = cv2.imread(file_path)
        
        # Inference
        results = model(img)[0]
        
        # Filter berdasarkan confidence
        confidence_threshold = 0.5
        bboxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy() 
        
        filtered_bboxes = []
        for bbox, score, cls in zip(bboxes, scores, classes):
            if score >= confidence_threshold:
                filtered_bboxes.append((bbox, score, cls))
        
        # Gambar Bounding Box
        for bbox, score, cls in filtered_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            class_name = model.names[int(cls)]
            color = CLASS_COLORS.get(class_name, (255, 255, 255))  # Fallback kalo salah nulis kelas (jadi putih)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {score:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        inferred_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        inferred_img_pil = Image.fromarray(inferred_img)
        inferred_img_pil = inferred_img_pil.resize((400, 400), Image.Resampling.LANCZOS)
        
        # Display hasil
        inferred_img_tk = ImageTk.PhotoImage(inferred_img_pil)
        self.inferred_label.configure(image=inferred_img_tk)
        self.inferred_label.image = inferred_img_tk

# Main Window
root = tk.Tk()
app = YOLOApp(root)
root.mainloop()