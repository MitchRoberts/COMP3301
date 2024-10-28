import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from PIL import Image
from PyQt5.QtWidgets import QApplication, QFileDialog
import math

class FilterProcessing:
    def __init__(self) -> None:
        #Prompt the user to select an image file
        self.image_array = self.load_image_via_dialog()

        #Copy image for future processing
        self.processed_img = self.image_array.copy()
        self.sigma = 10.0

        #Set up the figure and axes
        self.fig, (self.axes1, self.axes2) = plt.subplots(1, 2, figsize=(12, 5))

        #Adjust distance between subplots
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.25, wspace=0.3)

        #Display original image and placeholder for processed image, while checking for grayscale
        self.axes1.imshow(self.image_array, cmap='gray' if self.image_array.ndim == 2 else None)
        self.axes1.set_title('Original Image')
        self.axes2.imshow(self.image_array, cmap='gray' if self.image_array.ndim == 2 else None)
        self.axes2.set_title('Processed Image')

        #Disable axes for original and processed image
        self.axes1.axis('off')
        self.axes2.axis('off')

        #Create buttons and a text box
        self.setup_buttons()

        #Set up layout and show the plot
        plt.subplots_adjust(bottom=0.25)


    def load_image_via_dialog(self):
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

        #Check if the image is grayscale
        if img.mode == 'L':  
            return np.array(img) / 255.0  
        else:
            return np.array(img.convert('RGB')) / 255.0  

    def setup_buttons(self):
        """
        Setup buttons for different image processing tasks.
        """
        #Add noise button
        but_noise = plt.axes([0.05, 0.01, 0.10, 0.05])
        button_noise = Button(but_noise, "Add Noise")
        button_noise.on_clicked(lambda event: self.add_noise())

        #Triangle Filter button
        but_tri_filt = plt.axes([0.20, 0.01, 0.10, 0.05])
        button_triangle = Button(but_tri_filt, "Triangle Filter")
        button_triangle.on_clicked(lambda event: self.triangle_filter())

        #Median filter button
        but_med_filt = plt.axes([0.35, 0.01, 0.10, 0.05])
        button_median = Button(but_med_filt, "Median Filter")
        button_median.on_clicked(lambda event: self.median_filter())

        #Sigma value text input
        sigma_box_txt = plt.axes([0.45, 0.90, 0.1, 0.05])
        sigma_box = TextBox(sigma_box_txt, 'Sigma Value', initial="10")
        sigma_box.on_submit(self.update_sigma)

        #Gaussian filter button
        but_gaussian_filt = plt.axes([0.50, 0.01, 0.10, 0.05])
        button_gaussian = Button(but_gaussian_filt, "Gaussian Filter")
        button_gaussian.on_clicked(lambda event: self.gaussian_filter(self.processed_img, self.sigma))

        #Kuwahara filter button
        but_kuw_filt = plt.axes([0.65, 0.01, 0.10, 0.05])
        button_kuw = Button(but_kuw_filt, "Kuwahara Filter")
        button_kuw.on_clicked(lambda event: self.kuwahara_filter())

        #Mean filter button
        but_mean_filt = plt.axes([0.80, 0.01, 0.10, 0.05])
        button_mean = Button(but_mean_filt, "Mean Filter")
        button_mean.on_clicked(lambda event: self.mean_filter())

        plt.show()

    def update_sigma(self, text):
        """
        Update the sigma value based on the user input.
        """
        self.sigma = float(text)
        print(f"Sigma value updated to: {self.sigma}")
        return self.sigma
        

    def horizontal_filter(self, image, kernel, k_sum, w):
        """
        Function to apply a horizontal filter with padding.
        Works for both grayscale and RGB images.
        """

        #Get image dimensions depending on image type
        if image.ndim == 2:
            height, width = image.shape
            channels = 1
        elif image.ndim == 3:
            height, width, channels = image.shape
        else:
            raise ValueError("Unsupported image format")

        if channels == 1:
            #Pad only the width 
            padded_img = np.pad(image, ((0, 0), (w, w)), mode='edge')
        else:
            #Pad the width and keep the color channels for RGB
            padded_img = np.pad(image, ((0, 0), (w, w), (0, 0)), mode='edge')

        #Initialize output buffer
        T = np.zeros_like(image)

        #Loop through each row
        for q in range(height):
            for c in range(channels):
                if channels == 1:
                    for p in range(width):
                        h_sum = np.sum(padded_img[q, p:p + 2 * w + 1] * kernel)
                        T[q, p] = h_sum / k_sum
                else:
                    for p in range(width):
                        h_sum = np.sum(padded_img[q, p:p + 2 * w + 1, c] * kernel)
                        T[q, p, c] = h_sum / k_sum

        return T

    
    def vertical_filter(self, T, kernel, k_sum, w):
        """
        Function to apply a vertical filter with padding.
        Works for both grayscale and RGB images.
        """

        #Get image dimensions depending on image type
        if T.ndim == 2: 
            height, width = T.shape
            channels = 1
        elif T.ndim == 3:  
            height, width, channels = T.shape

        if channels == 1: 
            #Pad only the width
            padded_img = np.pad(T, ((w, w), (0, 0)), mode='edge')
        else:
            #Pad the width and keep the color channels for RGB
            padded_img = np.pad(T, ((w, w), (0, 0), (0, 0)), mode='edge')


        #Initialize output buffer
        V = np.zeros_like(T)

        #Loop through each column
        for p in range(width):
            for c in range(channels):
                if channels == 1:
                    for q in range(height):
                        v_sum = np.sum(padded_img[q:q + 2 * w + 1, p] * kernel)
                        V[q, p] = v_sum / k_sum
                else:
                    for q in range(height):
                        v_sum = np.sum(padded_img[q:q + 2 * w + 1, p, c] * kernel)
                        V[q, p, c] = v_sum / k_sum

        return V
    
    def add_noise(self):
        """
        Add Gaussian Noise to image
        """
        #Generate noise and apply to image, then clip
        noise = np.random.normal(0, 0.05, self.processed_img.shape)
        img_n = self.processed_img + noise
        img_n = np.clip(img_n, 0, 1)

        #Reassign new noisey image to processed_image
        self.processed_img = img_n

        #Update display
        if self.processed_img.ndim == 2:
            self.axes2.imshow(self.processed_img, cmap='gray')
        else:
            self.axes2.imshow(self.processed_img)
            
        plt.draw() 


    def gaussian_filter(self, image, sigma):
        """
        Apply a Gaussian filter to the image with a specified sigma.
        """
        def g_kernal(w, sigma):
            """
            Helper function to generate a Gaussian Kernel
            """
            kernel = [math.exp(-i**2 / (2 * sigma**2)) for i in range(-w, w + 1)]
            return np.array(kernel) / np.sum(kernel) 

        #Create horizontal and vertical filters
        w = 2 
        kernel_h = g_kernal(w, sigma)
        kernel_v = g_kernal(w, sigma)

        #Apply horizontal filter, then vertical
        T = self.horizontal_filter(image, kernel_h, np.sum(kernel_h), w)
        result = self.vertical_filter(T, kernel_v, np.sum(kernel_v), w)

        #Save the processed image
        self.processed_img = result

        #Update the display
        if self.processed_img.ndim == 2:
            self.axes2.imshow(self.processed_img, cmap='gray')
        else:
            self.axes2.imshow(self.processed_img)

        plt.draw() 

    def triangle_filter(self):
        """
        Apply triangle filter to image
        """
        #Triangle filter
        kernel = np.array([1, 2, 3, 2, 1]) / 9

        #Apply the horizontal and vertical filters
        T = self.horizontal_filter(self.processed_img, kernel, np.sum(kernel), 2)
        result = self.vertical_filter(T, kernel, np.sum(kernel), 2)

        #Save the processed image
        self.processed_img = result

        #Update the display
        if self.processed_img.ndim == 2:
            self.axes2.imshow(self.processed_img, cmap='gray')
        else:
            self.axes2.imshow(self.processed_img)
            
        plt.draw() 

    def median_filter(self):
        """
        Functiion to apply median filter
        """
        #Assign image variable for easier readability
        image = self.processed_img

        w = 2

        if image.ndim == 2:
            padded_img = np.pad(image, ((w, w), (w, w)), mode='edge')
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    result[i, j] = np.median(padded_img[i:i + 2 * w + 1, j:j + 2 * w + 1])
        else:
            padded_img = np.pad(image, ((w, w), (w, w), (0, 0)), mode='edge')
            result = np.zeros_like(image)
            for c in range(3):
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        result[i, j, c] = np.median(padded_img[i:i + 2 * w + 1, j:j + 2 * w + 1, c])

        #Save the processed image
        self.processed_img = result

        #Update the display
        if self.processed_img.ndim == 2:
            self.axes2.imshow(self.processed_img, cmap='gray')
        else:
            self.axes2.imshow(self.processed_img)
            
        plt.draw() 

    def kuwahara_filter(self):
        """
        Apply the Kuwahara filter
        """

        #Assign image variable for easier readability
        image = self.processed_img
        result = np.zeros_like(image)
        w = 2

        if image.ndim == 2:
            #Pad the image
            padded_img = np.pad(image, ((w, w), (w, w)), mode='edge')
            result = np.zeros_like(image)

            #Loop over pixels, define the overlapping regions
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    regions = []
                    for r in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                        regions.append(padded_img[i + r[0]:i + r[0] + 3, j + r[1]:j + r[1] + 3])

                    #Mean and variance for each region
                    mean = [np.mean(region) for region in regions]
                    var = [np.var(region) for region in regions]

                    #Find the region with the smallest variance and set mean
                    min_var = np.argmin(var)
                    result[i, j] = mean[min_var]

        else:
            #Pad the image
            padded_img = np.pad(image, ((w, w), (w, w), (0, 0)), mode='edge')
            result = np.zeros_like(image)

            #Loop over each color channel, define the overlapping regions
            for c in range(3):
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        regions = []
                        for r in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                            regions.append(padded_img[i + r[0]:i + r[0] + 3, j + r[1]:j + r[1] + 3, c])

                        #Mean and variance for each region
                        mean = [np.mean(region) for region in regions]
                        var = [np.var(region) for region in regions]

                        #Find the region with the smallest variance and set mean
                        min_var = np.argmin(var)
                        result[i, j, c] = mean[min_var]

        #Save the processed image
        self.processed_img = result

        #Update the display with the processed image
        if self.processed_img.ndim == 2:
            self.axes2.imshow(self.processed_img, cmap='gray')
        else:
            self.axes2.imshow(self.processed_img)
            
        plt.draw() 

    def mean_filter(self):
        """
        Function for applying mean filter
        """

        #Assign image variable for easier readability
        image = self.processed_img
        w = 2

        #Kernel for mean filter
        kernel = np.array([1, 1, 1, 1, 1]) / 5

        T = self.horizontal_filter(image, kernel, np.sum(kernel), w)
        result = self.vertical_filter(T, kernel, np.sum(kernel), w)

        #Save the processed image
        self.processed_img = result

        #Update the display with the processed image
        if self.processed_img.ndim == 2:
            self.axes2.imshow(self.processed_img, cmap='gray')
        else:
            self.axes2.imshow(self.processed_img)
            
        plt.draw() 

if __name__ == "__main__":
    test = FilterProcessing()