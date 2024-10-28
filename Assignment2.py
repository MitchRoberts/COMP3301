import numpy as np
from tkinter import Tk, Button, Label, Entry, Canvas, Frame
from PIL import Image, ImageTk
import sys

from PyQt5.QtWidgets import QApplication, QFileDialog

class FilterImage():

    def __init__(self, root) -> None:
        
        #Assigning main tkinter window
        self.root = root
        self.root.title("Image Processing")

        #Proper grid setup
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        #Canvas for original image with noise
        image_frame = Frame(self.root)
        image_frame.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        #Canvas for displaying image on the left
        self.canvas_left = Canvas(self.root, width=300, height=300)
        self.canvas_left.grid(row=0, column=0, padx=1, pady=10)

        #Canvas for displaying image on right
        self.canvas_right = Canvas(self.root, width=300, height=300)
        self.canvas_right.grid(row=0, column=1, padx=1, pady=10)

        self.current_image = self.load_image_via_explorer()

        self.display_image(self.current_image, "left")

        self.setp_buttons()

    def setp_buttons(self):
        """
        Setup buttons for different image processing tasks.
        """

        # Set button width and height for uniformity
        button_width = 15
        button_height = 2

        # Adjust grid column weights to ensure even distribution
        for col in range(7):  # Total 7 columns for buttons and sigma entry
            self.root.grid_columnconfigure(col, weight=1)

        # Add Noise button
        add_noise_button = Button(self.root, text="Add noise", width=button_width, height=button_height, command=self.add_noise)
        add_noise_button.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        # 5x5 Triangle button
        triangle_button = Button(self.root, text="5x5 Triangle", width=button_width, height=button_height, command=self.triangle_filter)
        triangle_button.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Sigma Label and Entry - Span across 2 columns for more space
        sigma_frame = Frame(self.root)
        sigma_frame.grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky="ew")

        sigma_label = Label(sigma_frame, text="Sigma:")
        sigma_label.pack(side="left", padx=5)

        self.sigma_entry = Entry(sigma_frame, width=10)
        self.sigma_entry.pack(side="left", padx=5)

        # 5x5 Gaussian button
        gaussian_button = Button(self.root, text="5x5 Gaussian", width=button_width, height=button_height, command=self.gaussian_filter)
        gaussian_button.grid(row=2, column=4, padx=5, pady=5, sticky="ew")

        # 5x5 Median button
        median_button = Button(self.root, text="5x5 Median", width=button_width, height=button_height, command=self.median_filter)
        median_button.grid(row=2, column=5, padx=5, pady=5, sticky="ew")

        # 5x5 Kuwahara button
        kuwahara_button = Button(self.root, text="5x5 Kuwahara", width=button_width, height=button_height, command=self.kuwahara_filter)
        kuwahara_button.grid(row=2, column=6, padx=5, pady=5, sticky="ew")

        # Configure row height to stretch proportionally
        self.root.grid_rowconfigure(2, weight=1)


    def display_image(self, image_array, side):
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(image_array)
        imgtk = ImageTk.PhotoImage(image=img)

        if side == "left":
            self.canvas_left.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas_left.image = imgtk
        elif side == "right":
            self.canvas_right.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas_right.image = imgtk

    def load_image_via_explorer(self):
        """
        Dynamically select an image via a file explorer to create histograms for
        """
        #Start PyQT file explorer and let user select image
        f_explorer = QApplication(sys.argv)
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Select an Image", "", "Image Files (*.png *.jpg *.jpeg)", options=options)

        if not file_path:
            #If no file is selected, raise error
            raise ValueError("No image file selected!")
        return self.load_image(file_path)
    
    def load_image(self, path):
        """
        Load image from path and return as a numpy array
        """
        #Open image and convert to numpy array
        img = Image.open(path)

        #Check if the image is grayscale, then preform normalization accordingly
        if img.mode == 'L':  #Grayscale mode
            return np.array(img) #/ 255.0  #Normalize grayscale image
        else:
            return np.array(img.convert('RGB')) #/ 255.0  #Normalize RGB image

    
    def convolution(self, image, kernel):
        """
        Method for convolution of kernal and image
        """

        image_height, image_width = image.shape

    def add_noise(self):
        """
        Add Gaussian noise to image
        """
        #Retirves value from GUI
        sigma = float(self.sigma_entry.get())

        #If value is 0/negative, add no noise
        if sigma <= 0:
            return

        #Apply Gaussian noise and add to image, then clip it to ensure values remain between 0,1
        #then update current image and diaplay on right

        noise = np.random.normal(0, sigma, self.current_image.shape)
        noisy_image = self.current_image + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        self.current_image = noisy_image
        self.display_image(self.current_image, "right")

    def triangle_filter(self):
        """
        Wrapper to apply horizontal and vertial filter
        """

        #w=2 for 5x5
        w = 2

        #Triangle filter weights
        triangle = np.array([1, 2, 3, 2, 1])
        k_sum = np.sum(triangle)

        #Intermediate image with only horizontal
        temp = self.horizontal_filter(self.current_image, triangle, w)

        #Apply vertical filter
        filtered_image = self.vertical_filter(temp, triangle, w)

        #Call to display image function to show image
        self.display_image(filtered_image)

    def gaussian_filter(self):
        """
        Wrapper to apply horizontal and vertial filter
        """

        #w=2 for 5x5
        w = 2

        #Triangle filter weights
        gaussian = np.array([1, 4, 6, 4, 1])
        k_sum = np.sum(gaussian)

        #Intermediate image with only horizontal
        temp = self.horizontal_filter(self.current_image, gaussian, k_sum, w)

        #Apply vertical filter
        filtered_image = self.vertical_filter(temp, gaussian, w)

        #Call to display image function to show image
        self.display_image(filtered_image)

    def horizontal_filter(self, image, kernal, k_sum, w):
        """
        Function to apply horizontal filter
        """

        #Assign height and width, and assign T as a temp buffer
        height, width = image.shape
        T = np.zeros_like(image)

        for q in range(height):
            #Loop throuhg each row, compute sum of first 2 * w + 1 row, q represents rows, p represents columns
            #then compute average for fist window and store in T
            h_sum = np.sum(image[q, :2 * w + 1] * kernal)
            T[q,w] = h_sum / k_sum

            #Update rest of row
            for p in range(w + 1, width -w):
                #Update h_sum by adding pixel entering window,
                #and suptracting pixel leaving, then store average in T
                h_sum += (image[q, p + w] * kernal[-1]) - (image[q, p - w - 1] * kernal[0])
                T[q,p] = h_sum / k_sum

        return T
    
    def vertical_filter(self, T, kernal, k_sum, w):
        """
        Function to apply vertical filter
        """

        #Assign height and width as dimensions of T, and assign G as a temp buffer
        height, width = T.shape
        V = np.zeros_like(T)

        for p in range(width):
            #Loop throuhg each column,
            v_sum = np.sum(T[:2 * w + 1, p])
            V[w,p] = v_sum / (2 * w + 1)

            for q in range(w + 1, height - w):
                v_sum += T[q + w, p] - T[q - w - 1, p]
                V[q,p] = v_sum / (2 * w + 1)

        return V
    
    def median_filter(self):
        pass

    def kuwahara_filter(self):
        pass

if __name__ == "__main__":
    root = Tk()
    test = FilterImage(root)
    root.mainloop()