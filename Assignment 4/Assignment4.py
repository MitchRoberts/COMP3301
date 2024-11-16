import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from PIL import Image
from PyQt5.QtWidgets import QApplication, QFileDialog
import math
import cv2

class Assignment4:

    def __init__(self):

         #Prompt the user to select an image file
        self.image_array = self.load_image_via_dialog()

        self.f_transform = None

        #Copy image for future processing
        self.processed_img = self.image_array.copy()
        self.lpf = 30.0

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
        Setup buttons for different thresholding tasks as per assignment requirements.
        """

        #Grayscale Image button
        but_grayscale = plt.axes([0.05, 0.01, 0.15, 0.05])
        button_grayscale = Button(but_grayscale, "Grayscale Image")
        button_grayscale.on_clicked(lambda event: self.grayscale_img())

        #Fourier Transform button
        but_display_ft = plt.axes([0.25, 0.01, 0.15, 0.05])
        button_display_ft = Button(but_display_ft, "Fourier Transform")
        button_display_ft.on_clicked(lambda event: self.ft_and_display())

        btn_ifft = plt.axes([0.45, 0.01, 0.15, 0.05])
        btn_ifft_button = Button(btn_ifft, 'Inverse Transform')
        btn_ifft_button.on_clicked(lambda event: self.inverse_fourier_transform())

        #Button for applying noise removal
        btn_filter = plt.axes([0.75, 0.01, 0.15, 0.05])
        btn_filter_button = Button(btn_filter, 'Apply Filter')
        btn_filter_button.on_clicked(lambda event: self.apply_lpf())
        
        #LPF value text input 
        lpf_box_txt = plt.axes([0.05, 0.90, 0.15, 0.05])
        lps_box = TextBox(lpf_box_txt, 'LPF Value', initial="30")
        lps_box.on_submit(self.update_lpf)

        #Reset image button
        but_reset = plt.axes([0.75, 0.90, 0.15, 0.05])
        button_reset = Button(but_reset, "Reset Image")
        button_reset.on_clicked(lambda event: self.reset_image())

        plt.show()

    def update_display(self):
        """
        Function to update display
        """
        if self.processed_img.ndim == 2:
            self.axes2.imshow(self.processed_img, cmap='gray')
        else:
            self.axes2.imshow(self.processed_img)
        
        plt.draw() 

    def reset_image(self):
        """
        Function for resetting to original image
        """
        print("Reseting to original image...")
        self.processed_img = self.image_array.copy()

        #Update display
        self.update_display()

    def update_lpf(self, text):
        """
        Update lpf value from user input
        """
        self.lpf = float(text)
        print(f"LPF Value updated to {self.lpf}")
        return self.lpf
    
    def apply_lpf(self):
        """
        Applies a Low-Pass Filter (LPF) to the Fourier Transform without performing the IFFT.
        """
        if not hasattr(self, 'f_transform'):
            print("Perform Fourier Transform first!")
            return

        rows, cols = self.f_transform.shape
        center_row, center_col = rows // 2, cols // 2

        # Create LPF mask
        mask = np.zeros((rows, cols))
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - center_col)**2 + (y - center_row)**2 <= self.lpf**2
        mask[mask_area] = 1  # Retain low frequencies

        # Apply the mask
        self.filtered_transform = self.f_transform * mask

        # Update processed image to show the filtered Fourier transform
        filtered_magnitude = np.abs(self.filtered_transform)
        filtered_magnitude_log = np.log(filtered_magnitude + 1)
        filtered_magnitude_log = (filtered_magnitude_log - filtered_magnitude_log.min()) / (
            filtered_magnitude_log.max() - filtered_magnitude_log.min()
        )
        self.processed_img = filtered_magnitude_log
        self.update_display()


    def fft1d(self, x):
        """
        Function to preform the 1D fourier transform
        """
        N = len(x)
        if N <= 1:
            return x
        even = self.fft1d(x[::2])
        odd = self.fft1d(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([even + factor[:N // 2] * odd, even + factor[N // 2:] * odd])

    def i_fft1d(self, x):
        """
        Function to preform the 1D inverse fourier transform
        """
        N = len(x)
        if N <= 1:
            return x
        even = self.i_fft1d(x[::2])
        odd = self.i_fft1d(x[1::2])
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        return np.concatenate([even + factor[:N // 2] * odd, even + factor[N // 2:] * odd]) / N

    def fft2d(self, image):
        """
        Function to apply the 2D fourier transform using 1D functions
        """
        
        #Apply 1D FT on each row
        rows_ft = np.array([self.fft1d(row) for row in image])
        
        #Apply 1D FT on each column
        cols_ft = np.array([self.fft1d(col) for col in rows_ft.T]).T
        return cols_ft

    def i_fft2d(self, image_ft):
        """
        Function to apply the 2D inverse fourier transform using 1D functions
        """
        
        #Apply Inverse 1D IFT on each row
        rows_ift = np.array([self.i_fft1d(ft_row) for ft_row in image_ft])
        
        #Apply Inverse 1D IFT on each column
        cols_ifft = np.array([self.i_fft1d(ft_col) for ft_col in rows_ift.T]).T
        rows, cols = image_ft.shape
        #Return the real part for the image 
        return np.abs(cols_ifft) / (rows * cols)

    #Shift the zero frequency to the center manually
    def shift_frequency(self, f_transform):
        """
        Function to make sure fourier transform frequencies are proplery alligned
        (low freq in the middle, etc.)
        """
        rows, cols = f_transform.shape

        #Swap quadrants to shift the zero-frequency component to the center
        mid_row, mid_col = rows // 2, cols // 2

        shifted_f_transform = np.zeros_like(f_transform, dtype=f_transform.dtype)

        shifted_f_transform[:mid_row, :mid_col] = f_transform[mid_row:, mid_col:]
        shifted_f_transform[mid_row:, mid_col:] = f_transform[:mid_row, :mid_col]
        shifted_f_transform[:mid_row, mid_col:] = f_transform[mid_row:, :mid_col]
        shifted_f_transform[mid_row:, :mid_col] = f_transform[:mid_row, mid_col:]

        return shifted_f_transform

    
    def fourier_transform(self, image):
        """
        Handler function to apply FT to an image
        """
        f_transform = self.shift_frequency(self.fft2d(image))
        self.f_transform = True
        return f_transform
    
    def inverse_fourier_transform(self):
        """
        Performs the Inverse Fourier Transform (IFFT) on the filtered Fourier Transform.
        """
        if not hasattr(self, 'filtered_transform'):
            print("Apply LPF first!")
            return

        # Shift the frequencies back to the original position
        f_transform_unshifted = self.shift_frequency(self.filtered_transform)

        # Perform inverse Fourier Transform
        filtered_image = self.i_fft2d(f_transform_unshifted)

        # Normalize the filtered image for display
        filtered_image = np.abs(filtered_image)
        filtered_image = (filtered_image - np.min(filtered_image)) / (np.max(filtered_image) - np.min(filtered_image))

        # Update and display the filtered image
        self.processed_img = filtered_image
        self.update_display()

    
    def remove_noise(self, f_transform, lpf=30):
        rows, cols = f_transform.shape
        center_row, center_col = rows // 2, cols // 2

        #Create LPF mask
        mask = np.zeros((rows, cols))

        #Create mask
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - center_col)**2 + (y - center_row)**2 <= lpf**2
        mask[mask_area] = 1

        #Apply mask to shifted fourier transform
        f_filtered = f_transform * mask
        final_image = self.i_fft2d(f_filtered)

        return final_image

    def grayscale_img(self):

        if self.processed_img.ndim == 3: 
            #Extract the R, G, B channels
            R = self.processed_img[:, :, 0]
            G = self.processed_img[:, :, 1]
            B = self.processed_img[:, :, 2]
        
            #Apply the grayscale conversion formula
            self.processed_img = 0.2989 * R + 0.5870 * G + 0.1140 * B

            #Update display
            self.update_display() 
            
            return self.processed_img
        else:
            #If the image is already grayscale, return it as is
            return self.processed_img
    
    def ft_and_display(self):
        """
        Function that will compute the Fourier transform and display it.
        """
        #Convert image to grayscale if it's not already
        if self.processed_img.ndim == 3:
            self.grayscale_img()
        
        #Compute the 2D Fourier Transform of the grayscale image
        f_transform = self.fft2d(self.processed_img)
        
        #Shift the zero-frequency component to the center of the spectrum
        f_transform_shifted = self.shift_frequency(f_transform)
        
        #Compute the magnitude spectrum and apply logarithmic scaling
        magnitude = np.abs(f_transform_shifted)
        magnitude_log = np.log(magnitude + 1)  
        
        #Normalize for better visualization
        magnitude_log = (magnitude_log - magnitude_log.min()) / (magnitude_log.max() - magnitude_log.min())
        
        #Update processed image with the log-scaled magnitude and update f_transform image
        self.processed_img = magnitude_log
        self.f_transform = f_transform_shifted

        #Display the result
        self.update_display()

    def opencv_fft(self):
        img_float32 = np.float32(self.processed_img)
        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        
        # Shift the zero-frequency component to the center of the spectrum
        dft_shift = np.fft.fftshift(dft)
        
        # Compute the magnitude spectrum
        magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        
        # Apply logarithmic scaling for better visualization
        magnitude_spectrum_log = np.log(magnitude_spectrum + 1)  # Adding 1 to avoid log(0)
        
        # Normalize the result to fall between 0 and 1 for display
        magnitude_spectrum_log = cv2.normalize(magnitude_spectrum_log, None, 0, 1, cv2.NORM_MINMAX)
        
        self.processed_img = magnitude_spectrum_log
        self.update_display()




if __name__ == "__main__":
    test = Assignment4()