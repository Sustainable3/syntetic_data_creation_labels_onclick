import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import numpy as np
import os
import uuid
import math
import random

OUTPUT_BASE_DIR = "dataset_yolo_obb"
IMG_DIR = os.path.join(OUTPUT_BASE_DIR, "images")
LBL_DIR = os.path.join(OUTPUT_BASE_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)


def calculate_obb_corners(cx, cy, w, h, angle_deg):
    """
    Oblicza 4 narożniki obróconego prostokąta.
    cx, cy - środek obiektu (w pikselach)
    w, h - wymiary obiektu PRZED obrotem (w pikselach)
    angle_deg - kąt obrotu (stopnie)
    """
    angle_rad = math.radians(-angle_deg) # Minus dla zgodności z PIL (kierunek obrotu)
    
    # Połowy wymiarów
    dw = w / 2
    dh = h / 2
    
    # Wektory narożników względem środka (0,0)
    # Kolejność: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    corners_local = [
        (-dw, -dh),
        (dw, -dh),
        (dw, dh),
        (-dw, dh)
    ]
    
    rotated_corners = []
    for x, y in corners_local:
        # Wzór na obrót punktu 2D
        rx = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        ry = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        # Przesunięcie do globalnego środka
        rotated_corners.append((cx + rx, cy + ry))
        
    return rotated_corners

def normalize_obb(corners, img_w, img_h):
    """Normalizuje współrzędne narożników do zakresu 0-1"""
    norm_corners = []
    for x, y in corners:
        nx = max(0, min(img_w, x)) / img_w # Clip 0-width
        ny = max(0, min(img_h, y)) / img_h # Clip 0-height
        norm_corners.append((nx, ny))
    return norm_corners

def rotate_point(x, y, cx, cy, angle_rad):
    """Obraca pojedynczy punkt wokół zadanego środka (do augmentacji całego zdjęcia)"""
    tx, ty = x - cx, y - cy
    rx = tx * math.cos(angle_rad) - ty * math.sin(angle_rad)
    ry = tx * math.sin(angle_rad) + ty * math.cos(angle_rad)
    return rx + cx, ry + cy

def apply_noise_np(img, intensity=20):
    arr = np.array(img)
    noise = np.random.normal(0, intensity, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype('uint8')
    return Image.fromarray(noisy, mode=img.mode)

# --- GENERATOR WARIACJI (Augmentacja OBB) ---

def generate_dataset_variants(original_img, original_labels_obb, base_name):
    """
    original_labels_obb: lista list [(class_id, [(x1,y1), (x2,y2)...]), ...] (współrzędne znormalizowane)
    """
    w_img, h_img = original_img.size
    
    for i in range(1, 4):
        img_aug = original_img.copy()
        # Odtwarzamy listę etykiet (deep copy, bo będziemy modyfikować)
        labels_aug = []
        for cls, pts in original_labels_obb:
            labels_aug.append((cls, list(pts))) # Kopia punktów

        # 1. Obrót całego zdjęcia (wraz z etykietami OBB)
        if i in [1, 3]:
            angle = random.randint(-15, 15)
            if angle != 0:
                # expand=False, żeby zachować wymiary tła
                img_aug = img_aug.rotate(angle, resample=Image.BICUBIC, expand=False)
                
                angle_rad = math.radians(-angle)
                cx_img, cy_img = w_img / 2, h_img / 2
                
                new_labels = []
                for cls, points in labels_aug:
                    new_pts = []
                    for nx, ny in points:
                        # Denormalizacja -> Obrót -> Normalizacja
                        px, py = nx * w_img, ny * h_img
                        rx, ry = rotate_point(px, py, cx_img, cy_img, angle_rad)
                        # Clip (żeby nie wyszło poza zdjęcie)
                        rx = max(0, min(w_img, rx))
                        ry = max(0, min(h_img, ry))
                        new_pts.append((rx / w_img, ry / h_img))
                    new_labels.append((cls, new_pts))
                labels_aug = new_labels

        # 2. Crop (przycięcie)
        if i in [2, 3]:
            crop_p = 0.1
            cx = random.randint(0, int(w_img * crop_p))
            cy = random.randint(0, int(h_img * crop_p))
            cw = w_img - cx - random.randint(0, int(w_img * crop_p))
            ch = h_img - cy - random.randint(0, int(h_img * crop_p))
            
            img_aug = img_aug.crop((cx, cy, cx+cw, cy+ch))
            
            new_labels = []
            for cls, points in labels_aug:
                new_pts = []
                all_points_valid = True
                for nx, ny in points:
                    px, py = nx * w_img, ny * h_img
                    # Przesunięcie o wektor cropa
                    npx, npy = px - cx, py - cy
                    
                    # Sprawdzenie czy punkt jest w nowym kadrze (z marginesem)
                    # W OBB to trudniejsze, upraszczamy: clipujemy do krawędzi
                    npx = max(0, min(cw, npx))
                    npy = max(0, min(ch, npy))
                    
                    new_pts.append((npx / cw, npy / ch))
                
                new_labels.append((cls, new_pts))
            labels_aug = new_labels

        # 3. Szum
        if random.random() > 0.3:
            img_aug = apply_noise_np(img_aug, random.randint(10, 40))

        # Zapis
        f_name = f"{base_name}_aug{i}"
        img_aug.convert("RGB").save(os.path.join(IMG_DIR, f"{f_name}.jpg"), quality=95)
        
        with open(os.path.join(LBL_DIR, f"{f_name}.txt"), "w") as f:
            for cls, pts in labels_aug:
                # Format YOLOv8 OBB: class x1 y1 x2 y2 x3 y3 x4 y4
                coords = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in pts])
                f.write(f"{cls} {coords}\n")
                
        print(f"   -> Utworzono OBB wariant: {f_name}")

# --- APLIKACJA GUI ---

class YoloObbApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 OBB Data Generator (Oriented Bounding Boxes)")
        self.root.geometry("1280x800")

        self.base_img = None
        self.work_img = None
        self.overlay_img = None
        
        # Lista etykiet. Format: (class_id, [(nx1, ny1), (nx2, ny2), (nx3, ny3), (nx4, ny4)])
        self.labels_obb = [] 
        self.tk_preview = None

        self.setup_ui()

    def setup_ui(self):
        # Panel sterowania
        ctrl = tk.Frame(self.root, width=300, bg="#dddddd", padx=10, pady=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(ctrl, text="GENERATOR OBB", font=("Arial", 14, "bold"), bg="#dddddd").pack(pady=10)
        
        tk.Button(ctrl, text="1. Wczytaj TŁO", command=self.load_base, bg="white").pack(fill=tk.X, pady=5)
        tk.Button(ctrl, text="2. Wczytaj PANEL", command=self.load_overlay, bg="white").pack(fill=tk.X, pady=5)

        tk.Label(ctrl, text="--- Parametry Panelu ---", bg="#dddddd", font=("Arial", 10, "bold")).pack(pady=(15,5))
        
        tk.Label(ctrl, text="Skala", bg="#dddddd").pack(anchor="w")
        self.s_scale = tk.Scale(ctrl, from_=10, to=200, orient="horizontal", bg="#dddddd")
        self.s_scale.set(100)
        self.s_scale.pack(fill=tk.X)

        tk.Label(ctrl, text="Obrót (Kluczowe dla OBB)", bg="#dddddd", fg="red").pack(anchor="w")
        self.s_rot = tk.Scale(ctrl, from_=-180, to=180, orient="horizontal", bg="#dddddd")
        self.s_rot.set(0)
        self.s_rot.pack(fill=tk.X)

        tk.Label(ctrl, text="Jasność / Kontrast", bg="#dddddd").pack(anchor="w")
        self.s_bright = tk.Scale(ctrl, from_=0.5, to=1.5, resolution=0.1, orient="horizontal", label="Jasność", bg="#dddddd")
        self.s_bright.set(1.0)
        self.s_bright.pack(fill=tk.X)
        
        self.s_noise = tk.Scale(ctrl, from_=0, to=50, orient="horizontal", label="Szum", bg="#dddddd")
        self.s_noise.pack(fill=tk.X)
        
        tk.Button(ctrl, text="Reset Suwaków", command=self.reset_sliders).pack(fill=tk.X, pady=5)

        tk.Label(ctrl, text="--- Zapis ---", bg="#dddddd", font=("Arial", 10, "bold")).pack(pady=(15,5))
        self.btn_save = tk.Button(ctrl, text="ZAPISZ (OBB Format)", command=self.save_obb, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), state=tk.DISABLED)
        self.btn_save.pack(fill=tk.X, pady=10, ipady=5)
        
        tk.Button(ctrl, text="Wyczyść / Reset", command=self.reset_canvas).pack(fill=tk.X)
        
        self.lbl_status = tk.Label(ctrl, text="Gotowy", bg="#dddddd", fg="blue")
        self.lbl_status.pack(side=tk.BOTTOM, pady=10)

        # Canvas
        self.cv_frame = tk.Frame(self.root, bg="#333")
        self.cv_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.canvas = tk.Canvas(self.cv_frame, bg="#333", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_move)

    def reset_sliders(self):
        self.s_scale.set(100)
        self.s_rot.set(0)
        self.s_bright.set(1.0)
        self.s_noise.set(0)

    def load_base(self):
        p = filedialog.askopenfilename()
        if p:
            self.base_img = Image.open(p).convert("RGBA")
            self.reset_canvas()
            self.lbl_status.config(text="Wczytano tło")

    def load_overlay(self):
        p = filedialog.askopenfilename()
        if p:
            self.overlay_img = Image.open(p).convert("RGBA")
            self.lbl_status.config(text="Wczytano panel")

    def reset_canvas(self):
        if self.base_img:
            self.work_img = self.base_img.copy()
            self.labels_obb = []
            self.redraw()
            self.btn_save.config(state=tk.NORMAL)

    def get_modified_overlay(self):
        """Przygotowuje panel (skala, obrót, efekty)"""
        if not self.overlay_img: return None, 0, 0
        
        img = self.overlay_img.copy()
        
        # 1. Skala
        scale = self.s_scale.get() / 100.0
        # Zapisujemy wymiary PO skalowaniu, ale PRZED obrotem (ważne dla matematyki OBB)
        base_w = int(img.width * scale)
        base_h = int(img.height * scale)
        img = img.resize((base_w, base_h), Image.Resampling.LANCZOS)
        
        # 2. Efekty
        bright = self.s_bright.get()
        if bright != 1.0: img = ImageEnhance.Brightness(img).enhance(bright)
        
        noise = self.s_noise.get()
        if noise > 0: img = apply_noise_np(img, noise)
        
        # 3. Obrót (dla wyświetlania)
        rot = self.s_rot.get()
        # expand=True jest konieczne, żeby PIL nie uciął rogów przy wyświetlaniu
        img_rotated = img.rotate(rot, expand=True, resample=Image.BICUBIC)
        
        return img_rotated, base_w, base_h

    def on_move(self, event):
        if not self.work_img or not self.overlay_img: return
        
        processed_ov, _, _ = self.get_modified_overlay()
        self.tk_preview = ImageTk.PhotoImage(processed_ov)
        
        self.canvas.delete("ghost")
        self.canvas.create_image(event.x, event.y, image=self.tk_preview, tag="ghost")
        
        # Opcjonalnie: Rysowanie linii OBB dla podglądu (czerwona ramka)
        # To wymagałoby przeliczenia rogów w locie tutaj, dla prostoty zostawiam "ducha"

    def on_click(self, event):
        if not self.work_img or not self.overlay_img: return
        
        img_rotated, base_w, base_h = self.get_modified_overlay()
        
        # Środek kliknięcia
        cx, cy = event.x, event.y
        
        # 1. Wklejenie obrazka
        # Musimy wycentrować wklejany obrazek (który jest powiększony przez expand=True)
        paste_w, paste_h = img_rotated.size
        paste_x = int(cx - paste_w / 2)
        paste_y = int(cy - paste_h / 2)
        
        self.work_img.paste(img_rotated, (paste_x, paste_y), mask=img_rotated)
        
        # 2. Obliczenie matematyczne OBB (kluczowy moment)
        rot_deg = self.s_rot.get()
        
        # Używamy base_w, base_h (wymiary prostokąta), a nie paste_w/h (wymiary bounding boxa obrazka)
        corners = calculate_obb_corners(cx, cy, base_w, base_h, rot_deg)
        
        # Normalizacja
        img_w, img_h = self.work_img.size
        norm_corners = normalize_obb(corners, img_w, img_h)
        
        # Dodanie do listy (class_id=0)
        self.labels_obb.append((0, norm_corners))
        
        self.redraw()
        self.lbl_status.config(text=f"Dodano obiekt OBB. Razem: {len(self.labels_obb)}")

    def redraw(self):
        if not self.work_img: return
        self.tk_bg = ImageTk.PhotoImage(self.work_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_bg, anchor="nw")

    def save_obb(self):
        if not self.labels_obb: return
        
        base_name = f"obb_{uuid.uuid4().hex[:6]}"
        
        # 1. Zapis Oryginału
        self.work_img.convert("RGB").save(os.path.join(IMG_DIR, f"{base_name}.jpg"))
        
        # Zapis TXT (Format OBB: class x1 y1 x2 y2 x3 y3 x4 y4)
        with open(os.path.join(LBL_DIR, f"{base_name}.txt"), "w") as f:
            for cls, pts in self.labels_obb:
                # pts to lista [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                # flattenujemy ją do stringa
                coords = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in pts])
                f.write(f"{cls} {coords}\n")
        
        self.lbl_status.config(text="Generowanie wariantów...")
        self.root.update()
        
        # 2. Generowanie wariacji
        try:
            generate_dataset_variants(self.work_img, self.labels_obb, base_name)
            messagebox.showinfo("Sukces", "Zapisano dane w formacie YOLOv8 OBB!")
            self.reset_canvas()
            self.lbl_status.config(text="Gotowe.")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloObbApp(root)
    root.mainloop()