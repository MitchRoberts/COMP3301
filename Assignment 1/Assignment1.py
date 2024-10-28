import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from PIL import Image
from PyQt5.QtWidgets import QApplication, QFileDialog
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

class histogram_dev:
    def __init__(self):
        #Prompt the user to select an image file
        self.image_array = self.load_image_via_dialog()

        #Copy image for future processing
        self.processed_image_array = self.image_array.copy()
        self.cutoff_fraction = 10.0

        #Set up the figure and axes
        self.fig, (self.axes1, self.axes2, self.axes3) = plt.subplots(1, 3, figsize=(12, 5))

        #Adjust distance between subplots
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.25, wspace=0.3)

        #Display original image and placeholder for processed image, while checking for grayscale
        self.axes1.imshow(self.image_array, cmap='gray' if self.image_array.ndim == 2 else None)
        self.axes1.set_title('Original Image')
        self.axes3.imshow(self.image_array, cmap='gray' if self.image_array.ndim == 2 else None)
        self.axes3.set_title('Processed Image')

        #Disable axes for original and processed image
        self.axes1.axis('off')
        self.axes3.axis('off')

        #Placeholder for histogram
        self.axes2.set_title('Image Histogram')

        #Create buttons and a text box
        self.setup_buttons()

        #Set up layout and show the plot
        plt.subplots_adjust(bottom=0.25)

        #Display the initial histogram
        self.display_histogram()

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
        if img.mode == 'L':  #Grayscale mode
            return np.array(img) / 255.0  #Normalize grayscale image
        else:
            return np.array(img.convert('RGB')) / 255.0  #Normalize RGB image

    def display_histogram(self):
        """
        Display the histogram for each color channel based on the processed image.
        """
        print("Updating Histogram...")

        #Clear previous histogram
        self.axes2.clear()

        #Determine if image is grayscale or RGB
        if self.processed_image_array.ndim == 2:
            #Grayscale image
            hist, bins = np.histogram(self.processed_image_array.ravel(), bins=256, range=(0, 1), density=True)

            #Plot the histogram
            self.axes2.plot(bins[:-1], hist / 255, color='gray')  #Use gray for grayscale images
            
            #Set axis labels and titles
            self.axes2.set_title('Grayscale Histogram')
            self.axes2.set_xlim([0, 1])
            self.axes2.set_xlabel('Intensity')
            self.axes2.set_ylabel('Probability Density')

        else:
            #RGB image
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                #Calculate histogram for each color channel
                hist, bins = np.histogram(self.processed_image_array[:, :, i].ravel(), bins=256, range=(0, 1), density=True)

                #Plot the histogram
                self.axes2.plot(bins[:-1], hist / 255, color=color)
            
            #Set axis labels and titles
            self.axes2.set_title('RGB Histogram')
            self.axes2.set_xlim([0, 1])
            self.axes2.set_xlabel('Intensity')
            self.axes2.set_ylabel('Probability Density')

        #Redraw to make sure new histogram gets put there
        self.fig.canvas.draw_idle()

    def stretch_histogram(self, event):
        """
        Apply basic histogram stretching and update the display.
        """
        #Calculate min/max intensity value for each color channel
        if self.processed_image_array.ndim == 2:
            #Grayscale image
            min_val = self.image_array.min()
            max_val = self.image_array.max()

            #Apply stretching and clip values
            stretched_image = (self.image_array - min_val) / (max_val - min_val)
        else:
            #RGB image
            min_val = self.image_array.min(axis=(0, 1))
            max_val = self.image_array.max(axis=(0, 1))

            #Apply stretching and clip values
            stretched_image = (self.image_array - min_val) / (max_val - min_val)
        
        self.processed_image_array = np.clip(stretched_image, 0, 1)
       
        #Display processed image on third axes, check for grayscale
        self.axes3.imshow(self.processed_image_array, cmap='gray' if self.processed_image_array.ndim == 2 else None)
        self.axes3.set_title('Stretched Image')
        self.axes3.axis('off')

        #Update histogram after processing
        self.display_histogram()

    def aggressive_stretch_histogram(self, event):
        """
        Apply aggressive histogram stretching based on the cutoff fraction and update the display.
        """
        if self.processed_image_array.ndim == 2:
            #Calculate lower/upper percentiles, apply aggressive stretch for Grayscale
            cutoff_fraction = self.cutoff_fraction / 100.0
            low_percentile = np.percentile(self.image_array, cutoff_fraction * 100)
            high_percentile = np.percentile(self.image_array, (1 - cutoff_fraction) * 100)
            stretched_image = (self.image_array - low_percentile) / (high_percentile - low_percentile)
        else:
            #Calculate lower/upper percentiles, apply aggressive stretch for RGB
            cutoff_fraction = self.cutoff_fraction / 100.0
            low_percentile = np.percentile(self.image_array, cutoff_fraction * 100, axis=(0, 1))
            high_percentile = np.percentile(self.image_array, (1 - cutoff_fraction) * 100, axis=(0, 1))
            stretched_image = (self.image_array - low_percentile) / (high_percentile - low_percentile)
        
        #Clip values
        self.processed_image_array = np.clip(stretched_image, 0, 1)
        
        #Display processed image on third axes, check for grayscale
        self.axes3.imshow(self.processed_image_array, cmap='gray' if self.processed_image_array.ndim == 2 else None)
        self.axes3.set_title('Aggressive Stretched Image')
        self.axes3.axis('off')

        #Update histogram after processing
        self.display_histogram()

    def equalize_histogram(self, event):
        """
        Apply histogram equalization using the provided working example and update the display.
        """
        def histogram_equalization(image):
            """
            Perform histogram equalization on an RGB or grayscale image.
            """
            if image.ndim == 2:
                #Grayscale image
                hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 1), density=True)
                cdf = np.cumsum(hist)
                cdf = cdf / cdf[-1]
                equalized_image = np.interp(image.ravel(), np.linspace(0, 1, 256), cdf)
                return equalized_image.reshape(image.shape)
            else:
                #RGB image
                image_hsv = rgb_to_hsv(image)
                v_channel = image_hsv[:, :, 2]
                hist, _ = np.histogram(v_channel.ravel(), bins=256, range=(0, 1), density=True)
                cdf = np.cumsum(hist)
                cdf = cdf / cdf[-1]
                equalized_v = np.interp(v_channel.ravel(), np.linspace(0, 1, 256), cdf)
                equalized_v = equalized_v.reshape(v_channel.shape)
                image_hsv[:, :, 2] = equalized_v
                return hsv_to_rgb(image_hsv)

        #Apply equalization and update the processed image and display it, while checking for grayscale
        equalized_image = histogram_equalization(self.image_array)
        self.processed_image_array = equalized_image
        self.axes3.imshow(self.processed_image_array, cmap='gray' if self.processed_image_array.ndim == 2 else None)
        self.axes3.set_title('Equalized Image')
        self.axes3.axis('off')

        #Update the histogram after equalization
        self.display_histogram()

    def update_cutoff(self, text):
        """
        Update the cutoff fraction based on the user input.
        """
        try:
            #Update the cutoff value with user input
            self.cutoff_fraction = float(text)
            print(f"Cutoff fraction updated to: {self.cutoff_fraction}%")
        
        except ValueError:
            #Handling for invalid input
            print(f"{self.cutoff_fraction} is an invalid cutoff fraction")

    def setup_buttons(self):
        """
        Setup buttons for different image processing tasks.
        """
        #Display histogram button
        but_disp_hist = plt.axes([0.03, 0.01, 0.15, 0.05])
        button_disp = Button(but_disp_hist, "Display Histogram")
        button_disp.on_clicked(lambda event: self.display_histogram())

        #Stretch histogram button
        but_stretch_hist = plt.axes([0.23, 0.01, 0.15, 0.05])
        button_stretch = Button(but_stretch_hist, "Histogram Stretch")
        button_stretch.on_clicked(self.stretch_histogram)

        #Cutoff fraction text input
        ax_cutoff_box = plt.axes([0.1, 0.90, 0.1, 0.05])
        cutoff_box = TextBox(ax_cutoff_box, 'Cutoff fraction', initial="10")
        cutoff_box.on_submit(self.update_cutoff)

        #Aggressive stretch button
        but_agstretch_hist = plt.axes([0.63, 0.01, 0.15, 0.05])
        button_agstretch = Button(but_agstretch_hist, "Aggressive Stretch")
        button_agstretch.on_clicked(self.aggressive_stretch_histogram)

        #Equalize histogram button
        but_equal_hist = plt.axes([0.83, 0.01, 0.15, 0.05])
        button_equalize = Button(but_equal_hist, "Histogram Equalization")
        button_equalize.on_clicked(self.equalize_histogram)


        plt.show()

#Main execution
if __name__ == "__main__":
    test = histogram_dev()
