import cv2
from PIL import Image, ImageTk
import tkinter.font as tkFont
import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt


"""
    A simple test  code for  image processing application built using Tkinter and OpenCV.

    This application allows users to perform diffent kind operations on images, including:
    - Opening images from files
    - Selecting regions of interest (ROIs)
    - Cropping images based on selected ROIs
    - Normalizing cropped images
    - Displaying histograms of selected parts of images
    - Zooming in and out of images

    Attributes:
        project_airss (tk.Tk): The main Tkinter window of the application.
        image (numpy.ndarray): The currently loaded image as a NumPy array.
        normalized_image (PIL.Image.Image): The normalized version of the cropped image.
        photo_image (ImageTk.PhotoImage): The currently displayed image in the GUI.
        cropped_image (numpy.ndarray): The cropped region of the loaded image.
        selected_roi (tuple): Coordinates of the selected region of interest (x0, y0, x1, y1).
        box_start (tuple): Starting coordinates of the box for drawing ROI.
    """

class ImageProcessorApp:
    def __init__(self, project_airss):
        self.project_airss = project_airss
        self.project_airss.title("AiRISS")
        self.create_widgets()
        self.image = None
        self.normalized_image = None
        self.photo_image = None
        self.cropped_image = None
        bold_font = tkFont.Font(weight="bold")
        self.footer_label = tk.Label(project_airss, text="", justify=tk.RIGHT, font=bold_font)
        self.footer_label.pack(side=tk.BOTTOM, fill=tk.X)
        self.footer_label.configure(bg="gray", foreground="white")
        self.selected_roi = None
        self.box_start = None
      

    def create_widgets(self):
        """
        create_widget function is athe GUI widgets  for the application.
        This method initialize and configures various widgets such as menus, canvas, and scrollbars
        within the main Tkinter window.

        Returns:
                None
        """  
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

        

        self.canvas_frame = tk.Frame(self.project_airss)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.image_canvas = tk.Canvas(self.canvas_frame, cursor="cross")
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar_y = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_canvas.config(yscrollcommand=self.scrollbar_y.set)

        self.scrollbar_x = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.image_canvas.config(xscrollcommand=self.scrollbar_x.set)

        self.image_canvas.bind("<Motion>", self.update_footer)
   
    

    def open_image(self): 
        """
        Opens an image file dialog and loads the selected image into the application.
        This method prompts the user to select an image file using a file dialog 
        and it will open image unchnaged either its 8, 10, 12 ,16 bits. If a file
        is selected, it is loaded using OpenCV, converted to RGB format if necessary, and
        converted to a PIL format for displaying in the GUI.
         """
        
        file_path = filedialog.askopenfilename()
        if file_path:
            # Open image using OpenCV
            self.image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            conv_image = self.image
            # Convert image to 8 bits
            if self.image.dtype == "uint16":
                conv_image = cv2.convertScaleAbs(self.image, alpha=(255.0/65535.0))
            # Convert image to PIL format for displaying
            self.photo_image = ImageTk.PhotoImage(Image.fromarray(conv_image))
            self.display_image(self.photo_image)
            self.update_footer()




    def display_image(self, img):
        '''    
        dispaly image function to  Display the provided image on the canvan
        and  Adjust the scroll region of the canvas based on the bounding box of all items
        '''
        self.image_canvas.create_image(0, 0, image=img, anchor=tk.NW)
     
     # Adjust the scroll region of the canvas based on the bounding box of all items
        self.image_canvas.config(scrollregion=self.image_canvas.bbox(tk.ALL))
 

    def start_box(self, event):
        """
        Initializes the selection of a bounding box.

        This function is triggered when the left mouse button is pressed.
        It records the starting position of the bounding box.

        Parameters:
            event (tk.Event): The mouse event that triggered the function call.

        Returns:
            None
        """   
        self.box_start = (event.x, event.y)
        
        

    def draw_box(self, event):
        """
        Draws the bounding box on the canvas.

        This function is called when the mouse is moved after the left mouse button is pressed.
        It draws the bounding box on the canvas as the mouse is dragged.

        Parameters:
            event (tk.Event): The mouse event that triggered the function call.

        Returns:
            None
        """
        if self.box_start:
            x0, y0 = self.box_start
            x1, y1 = event.x, event.y
            self.image_canvas.delete("box")
            self.image_canvas.create_rectangle(x0, y0, x1, y1, outline="red", tags="box")
    

    def end_box(self, event):
        """
        Finalizes the selection of the bounding box.

        This function is called when the left mouse button is released.
        It completes the selection of the bounding box by updating its coordinates
        and unbinding mouse events to prevent further modifications.

        Parameters:
            event (tk.Event): The mouse event that triggered the function call.

        Returns:
            None
        """
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
  
    def select_roi(self):
        
        '''
        Initiates the process of selecting a region of interest (ROI) on the canvas.
        This function binds mouse events to start, draw, and finalize the selection of a bounding box.
        When this method is called, users can start selecting a bounding box by clicking and dragging the mouse.

        Returns:
            None
        '''
        self.image_canvas.bind("<Button-1>", self.start_box)
        self.image_canvas.bind("<B1-Motion>", self.draw_box)
        self.image_canvas.bind("<ButtonRelease-1>", self.end_box)       


    def crop_image(self):     
        ''' 
        crop_image function to crop the image based on the selected roi
        then display it with canvas
        '''
        
        if self.selected_roi and self.image is not None:
            x0, y0, x1, y1 = self.selected_roi
            cropped_img = self.image[y0:y1, x0:x1]
            self.cropped_image = cropped_img
        else:
            messagebox.showerror("Error", "No ROI selected or no image loaded.")

    def display(self, cropped_img, titel):
        cropped_window = tk.Toplevel(self.project_airss)
        cropped_window.title(titel)

        # Convert the cropped image to photoimage
        cropped_photo = ImageTk.PhotoImage(cropped_img)

        # Display the cropped image 
        cropped_label = tk.Label(cropped_window, image=cropped_photo)
        cropped_label.pack()

        # Update the cropped image reference to avoid garbage collected
        cropped_label.image = cropped_photo

     

     
    def normalize_image(self):
            """
            Normalize the cropped image and display the normalized version.

            This function first checks if a cropped image is available.
            If available, it converts the cropped image to a NumPy array and checks if it's in grayscale.
            Then, it normalizes the image using OpenCV's normalize function.
            After normalization, it converts the normalized NumPy array back to a PIL Image.
            The normalized image is stored for future reference and displayed on the canvas.
            If no cropped image is available, an error message is displayed.

            Returns:
                None
            """   
            if self.cropped_image is not None:
                # Convert PIL 
                cropped_image_np = np.array(self.cropped_image)
              
                if len(cropped_image_np.shape) == 3 and cropped_image_np.shape[2] == 3:
                    cropped_image_np = cv2.cvtColor(cropped_image_np, cv2.COLOR_RGB2GRAY)
                # Normalize the image
                norm_img = cv2.normalize(cropped_image_np, None, 0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # Convert normalized image back to PIL Image
                norm_img_pil = Image.fromarray((norm_img * 255).astype(np.uint8))
                self.normalized_image = norm_img_pil  # Store normalized image
                # Display the normalized image
                self.display(norm_img_pil, "Normalized Image")
            else:
                messagebox.showerror("Error", "No cropped image or imageloaded.")
           

    def save_image(self):
            """
            Normalize the cropped image and display the normalized version.
            This function first checks if a cropped image is available.
            If available, it converts the cropped image to a NumPy array and checks if it's in grayscale.
            Then, it normalizes the image using OpenCV's normalize function.
            After normalization, it converts the normalized NumPy array back to a PIL Image.
            The normalized image is stored for future reference and displayed on the canvas.

            If no cropped image is available, an error message is displayed.
            """    
            
            if self.normalized_image is not None:  # Check if normalized image is available
                file_path = filedialog.asksaveasfilename(defaultextension=".png")
                if file_path:
                    # Save normalized image using PIL
                    self.normalized_image.save(file_path)
            else:
                messagebox.showerror("Error", "no normalized image tobe saved.")


    def show_histogram(self):
        '''
        Display the histogram for the selected region of interest (ROI).
        This function calculates the histogram for each color channel of the cropped image.
        It then plots the histograms using Matplotlib.If no ROI is selected or no image is loaded,
        an error message is displayed.
        Returns:
        None
        '''
        
        if self.selected_roi and self.image is not None:
            

            # Calculate histograms for each color channel
            color = ('r', 'g', 'b')
            max_pixel_value = 2 ** (self.cropped_image.dtype.itemsize * 8)
            for i, col in enumerate(color):
                channel = self.cropped_image[:, :, i] if len(self.cropped_image.shape) == 3 else self.cropped_image
                histogram = cv2.calcHist([channel], [0], None, [max_pixel_value], [0, max_pixel_value])
                plt.plot(histogram, color=col)

            plt.xlim([0, max_pixel_value])
            plt.title("Histogram for Selected Part of Image")
            plt.xlabel("Pixel value")
            plt.ylabel("Frequency")
            plt.show()

        else:
            messagebox.showerror("Error", "No ROI selected or no image loaded.")
         

    def zoom_image(self, zoom_in=True):
        '''
        Zoom in or out on the displayed image.

        This function resizes the image based on the zoom factor (1.25 for zooming in, 0.8 for zooming out).
        It converts the image to the appropriate format for displaying using PIL and updates the canvas.

        Parameters:
        zoom_in (bool): A flag indicating whether to zoom in (True) or out (False). Default is True for zooming in.

        Returns:
            None
        '''
        
        y, x, _ = self.image.shape
        factor = 1.25 if zoom_in else 0.8
        new_size = (int(x * factor), int(y * factor))
        # self.photo_image = ImageTk.PhotoImage(Image.fromarray(conv_image))
        # self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)
        self.image = cv2.resize(self.image, (new_size[0], new_size[1]), interpolation=cv2.INTER_LANCZOS4)
        conv_image = self.image
        # Convert image to RGB format
        if self.image.dtype == "uint16":
            conv_image = cv2.convertScaleAbs(self.image, alpha=(255.0/65535.0))
        # Convert image to PIL format for displaying
        self.photo_image = ImageTk.PhotoImage(Image.fromarray(conv_image))
        self.display_image(self.photo_image)
   
        
    def update_footer(self, event=None):
        ''' 
        Update the footer label with information about the image.
        This function updates the footer label with details such as image size, type, 
        mouse position,and pixel value.
        '''
        if self.image is not None:
            x, y, _ = self.image.shape
            bits = self.image.dtype
            mouse_pos = f"X: {event.x} Y: {event.y}" if event else "N/A"
            pixel_val = "N/A"
            if event:
                try:
                    pixel_val = self.image[event.y, event.x]
                except:
                    pixel_val = "Out of bounds"
            footer_text = f"Image size: {x} x {y}  Image_type: {bits}  Mouse position: {mouse_pos}  Pixel value: {pixel_val}"
            self.footer_label.config(text=footer_text)


if __name__ == "__main__":
    root = tk.Tk()
    #geomtry to resize the gui as needed
    
    root.geometry("900x900+400+100")
    app = ImageProcessorApp(root)
    root.mainloop()
