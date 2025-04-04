import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

class NutWasherDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Nut and Washer Detector")
        
        # Main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons
        ttk.Button(self.main_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, pady=5)
        ttk.Button(self.main_frame, text="Process Image", command=self.process_image).grid(row=0, column=1, pady=5)
        
        # Circularity slider
        self.circularity_label = ttk.Label(self.main_frame, text="Washer Circularity Threshold:")
        self.circularity_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.circularity_var = tk.DoubleVar(value=0.87)
        self.circularity_slider = ttk.Scale(
            self.main_frame, 
            from_=0.5, 
            to=1.0, 
            orient=tk.HORIZONTAL, 
            length=200, 
            variable=self.circularity_var,
            command=self.update_circularity_label
        )
        self.circularity_slider.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        self.circularity_value_label = ttk.Label(self.main_frame, text="0.87")
        self.circularity_value_label.grid(row=1, column=2, sticky=tk.W, pady=5)
        
        # Image display
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Results display
        self.result_text = tk.Text(self.main_frame, height=5, width=40)
        self.result_text.grid(row=3, column=0, columnspan=3, pady=5)
        
        self.current_image = None
        self.processed_image = None

    def update_circularity_label(self, value):
        """Update the label to show current circularity threshold"""
        formatted_value = "{:.2f}".format(float(value))
        self.circularity_value_label.config(text=formatted_value)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image)

    def detect_nuts_and_washers(self, image):
        
        circularity_threshold = self.circularity_var.get()

        
        # Convert to grayscale and preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        blurred = cv2.medianBlur(gray, 7)


        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 7, 1
        )

        # Morphological operations to reduce noise and shadows
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours and hierarchy
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        nut_count = 0
        washer_count = 0
        processed_image = image.copy()

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Filter by area
            if area < 10 or area > 100000:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

            # Bounding box and aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # Check for nuts (hexagonal shape with inner circle)
            if len(approx) == 6 and 0.8 <= aspect_ratio <= 1.2:  # Hexagonal outer shape
                # Check for inner hole (child contour)
                has_inner_circle = False
                for j, h in enumerate(hierarchy[0]):
                    if h[3] == i:  # Child contour
                        inner_area = cv2.contourArea(contours[j])
                        inner_perimeter = cv2.arcLength(contours[j], True)
                        circularity = (
                            4 * np.pi * inner_area / (inner_perimeter ** 2)
                            if inner_perimeter > 0
                            else 0
                        )
                        if inner_area > 10 and circularity > 0.7:  # Confirm circular hole
                            has_inner_circle = True
                            break

                if has_inner_circle:
                    cv2.drawContours(processed_image, [contour], -1, (0, 0, 255), 2)  # Red
                    nut_count += 1
                    continue

            # Check for washers (circular with inner hole)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            if circularity > circularity_threshold and 0.9 <= aspect_ratio <= 1.1:
                # Verify inner hole (child contour)
                has_inner_hole = False
                for j, h in enumerate(hierarchy[0]):
                    if h[3] == i:  # Child contour
                        inner_area = cv2.contourArea(contours[j])
                        inner_perimeter = cv2.arcLength(contours[j], True)
                        inner_circularity = (
                            4 * np.pi * inner_area / (inner_perimeter ** 2)
                            if inner_perimeter > 0
                            else 0
                        )
                        if inner_area > 10 and inner_circularity > 0.8:  # Confirm circular hole
                            has_inner_hole = True
                            break

                if has_inner_hole:
                    cv2.drawContours(processed_image, [contour], -1, (0, 255, 0), 2)  # Green
                    washer_count += 1

        return processed_image, nut_count, washer_count





    def process_image(self):
        if self.current_image is None:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Please load an image first!")
            return

        self.processed_image, nut_count, washer_count = self.detect_nuts_and_washers(self.current_image)
        self.display_image(self.processed_image)
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Detection Results:\nNuts: {nut_count}\nWashers: {washer_count}")

    def display_image(self, cv_image):
        if cv_image is None:
            return
            
        # Resize image to fit display
        max_size = 500
        height, width = cv_image.shape[:2]
        
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv_image = cv2.resize(cv_image, (new_width, new_height))
            
        # Convert CV2 image to PhotoImage
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        
        # Update label
        self.image_label.configure(image=photo)
        self.image_label.image = photo

def main():
    root = tk.Tk()
    root.title("Nut and Washer Detector")
    app = NutWasherDetector(root)
    root.mainloop()

if __name__ == "__main__":  
    main()