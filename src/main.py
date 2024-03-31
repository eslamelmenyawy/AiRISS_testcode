import cv2
from PIL import Image, ImageTk
import tkinter.font as tkFont
import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt

class ImageProcessorApp:
    def __init__(self, project_airss):
        self.project_airss = project_airss
        self.project_airss.title("airiss")
        self.create_widgets()
        self.image = None
        self.photo_image = None
        self.cropped_image = None
        bold_font = tkFont.Font(weight="bold")
        self.footer_label = tk.Label(project_airss, text="", justify=tk.RIGHT, font=bold_font)
        self.footer_label.pack(side=tk.BOTTOM, fill=tk.X)
        self.footer_label.configure(bg="gray", foreground="white")
        self.selected_roi = None
        self.box_start = None

    def create_widgets(self):
        self.menu_bar = tk.Menu(self.project_airss)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Open", command=self.open_image)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Save", command=self.save_image)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.algorithm_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.algorithm_menu.add_command(label="Select ROI", command=self.select_roi)
        self.algorithm_menu.add_command(label="Crop", command=self.crop_image)
        self.algorithm_menu.add_command(label="Normalize", command=self.normalize_image)
        self.algorithm_menu.add_command(label="Histogram", command=self.show_histogram)
        self.menu_bar.add_cascade(label="Algorithm", menu=self.algorithm_menu)

        self.action_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.action_menu.add_command(label="Zoom In", command=lambda: self.zoom_image(True))
        self.action_menu.add_command(label="Zoom Out", command=lambda: self.zoom_image(False))
        self.menu_bar.add_cascade(label="Actions", menu=self.action_menu)

        self.project_airss.config(menu=self.menu_bar)

        self.image_canvas = tk.Canvas(self.project_airss, cursor="cross")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        self.image_canvas.bind("<Motion>", self.update_footer)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Open image using OpenCV
            self.image = cv2.imread(file_path)
            # Convert image to RGB format
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            # Convert image to PIL format for displaying
            self.image = Image.fromarray(self.image)
            self.photo_image = ImageTk.PhotoImage(self.image)
            self.display_image(self.photo_image)
            self.update_footer()

    def save_image(self):
        if self.image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if file_path:
                # Convert image from PIL format to OpenCV format
                image_cv = np.array(self.image)
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
                # Save image using OpenCV
                cv2.imwrite(file_path, image_cv)
        else:
            messagebox.showerror("Error", "No image to save.")

    def display_image(self, img):
        self.image_canvas.create_image(0, 0, image=img, anchor=tk.NW)
        self.image_canvas.config(scrollregion=self.image_canvas.bbox(tk.ALL))

    def start_box(self, event):
        self.box_start = (event.x, event.y)

    def draw_box(self, event):
        if self.box_start:
            x0, y0 = self.box_start
            x1, y1 = event.x, event.y
            self.image_canvas.delete("box")
            self.image_canvas.create_rectangle(x0, y0, x1, y1, outline="red", tags="box")

    def end_box(self, event):
        if self.box_start:
            x0, y0 = self.box_start
            x1, y1 = event.x, event.y
            # Ensure x1 is greater than x0 and y1 is greater than y0
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            self.selected_roi = (x0, y0, x1, y1)
            self.box_start = None
            self.image_canvas.unbind("<Button-1>")
            self.image_canvas.unbind("<B1-Motion>")
            self.image_canvas.unbind("<ButtonRelease-1>")



    def update_footer(self, event=None):
        if self.image:
            if isinstance(self.image, Image.Image):
                x, y = self.image.size
                bits = self.image.mode
            else:
                y, x, _ = self.image.shape
                bits = "RGB"
            mouse_pos = f"X: {event.x} Y: {event.y}" if event else "N/A"
            pixel_val = "N/A"
            if event and isinstance(self.image, Image.Image):
                try:
                    pixel_val = self.image.getpixel((event.x, event.y))
                except:
                    pixel_val = "Out of bounds"
            footer_text = f"Image_Size: {x}x{y}  Image_type:{bits}  Mouse_position:{mouse_pos}  Pixel_Value:{pixel_val}"
            self.footer_label.config(text=footer_text)


    def select_roi(self):
        self.image_canvas.bind("<Button-1>", self.start_box)
        self.image_canvas.bind("<B1-Motion>", self.draw_box)
        self.image_canvas.bind("<ButtonRelease-1>", self.end_box)

    def crop_image(self):
        if self.selected_roi and self.image:
            x0, y0, x1, y1 = self.selected_roi
            if isinstance(self.image, Image.Image):
                cropped_img = self.image.crop((x0, y0, x1, y1))
            else:
                cropped_img = self.image[y0:y1, x0:x1]
            # self.display(cropped_img, "Cropped Image")
            self.cropped_image = cropped_img
        else:
            messagebox.showerror("Error", "No ROI selected or no image loaded.")

    def display(self, cropped_img, titel):
        cropped_window = tk.Toplevel(self.project_airss)
        cropped_window.title(titel)

        # Convert the cropped image to PhotoImage
        cropped_photo = ImageTk.PhotoImage(cropped_img)

        # Display the cropped image in a label
        cropped_label = tk.Label(cropped_window, image=cropped_photo)
        cropped_label.pack()

        # Update the cropped image reference to prevent it from being garbage collected
        cropped_label.image = cropped_photo

    def normalize_image(self):
        if self.cropped_image is not None:
            # Convert PIL Image to OpenCV Image
            cropped_image_np = np.array(self.cropped_image)
            # Convert image to grayscale if it's not already
            if len(cropped_image_np.shape) == 3 and cropped_image_np.shape[2] == 3:
                cropped_image_np = cv2.cvtColor(cropped_image_np, cv2.COLOR_RGB2GRAY)
            # Normalize the image
            norm_img = cv2.normalize(cropped_image_np, None, 0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # Convert normalized image back to PIL Image
            norm_img_pil = Image.fromarray((norm_img * 255).astype(np.uint8))
            # Display the normalized image
            self.display(norm_img_pil, "Normalized Image")
        else:
            messagebox.showerror("Error", "No cropped image or image loaded.")

    def show_histogram(self):
        if self.selected_roi and self.image:
            x0, y0, x1, y1 = self.selected_roi
            selected_part = self.image.crop((x0, y0, x1, y1)) if isinstance(self.image, Image.Image) else self.image[y0:y1, x0:x1]
            selected_part_np = np.array(selected_part)

            # Check if selected part is not empty
            if selected_part_np.size == 0:
                messagebox.showerror("Error", "Selected part of the image is empty.")
                return

            # Calculate histograms for each color channel
            color = ('r', 'g', 'b')
            for i, col in enumerate(color):
                channel = selected_part_np[:, :, i] if len(selected_part_np.shape) == 3 else selected_part_np
                histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
                plt.plot(histogram, color=col)
                plt.xlim([0, 256])

            plt.title("Histogram for Selected Part of Image")
            plt.xlabel("Pixel value")
            plt.ylabel("Frequency")
            plt.show()

        else:
            messagebox.showerror("Error", "No ROI selected or no image loaded.")

    def zoom_image(self, zoom_in=True):
        if isinstance(self.image, Image.Image):
            x, y = self.image.size
        else:
            y, x, _ = self.image.shape
        factor = 1.25 if zoom_in else 0.8
        new_size = (int(x * factor), int(y * factor))
        if isinstance(self.image, Image.Image):
            self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(self.image)
            self.display_image(self.photo_image)
        else:
            print("erroe")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x900+400+100")
    app = ImageProcessorApp(root)
    root.mainloop()
