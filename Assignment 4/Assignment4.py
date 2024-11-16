import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from PIL import Image
from PyQt5.QtWidgets import QApplication, QFileDialog

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

        #Fourier Transform button
        but_display_ft = plt.axes([0.05, 0.01, 0.15, 0.05])
        button_display_ft = Button(but_display_ft, "Fourier Transform")
        button_display_ft.on_clicked(lambda event: self.ft_and_display())

        #Inverse fourier transform button
        but_ifft = plt.axes([0.25, 0.01, 0.15, 0.05])
        but_ifft_button = Button(but_ifft, 'Inverse Transform')
        but_ifft_button.on_clicked(lambda event: self.inverse_fourier_transform())

        #Button for applying noise removal with LPF
        but_filter = plt.axes([0.45, 0.01, 0.15, 0.05])
        but_filter_button = Button(but_filter, 'Apply Filter')
        but_filter_button.on_clicked(lambda event: self.apply_lpf())

        #Button for Sboel edge detection
        but_sobel = plt.axes([0.65, 0.01, 0.10, 0.05])
        but_sobel_button = Button(but_sobel, "Sobel")
        but_sobel_button.on_clicked(lambda event: self.sobel_edge())

        #Button for Cannyedge detection
        but_canny = plt.axes([0.85, 0.01, 0.10, 0.05])
        but_canny_button = Button(but_canny, "Canny")
        but_canny_button.on_clicked(lambda event: self.canny_edge_detection())
        
        #Grayscale Image button
        but_grayscale = plt.axes([0.43, 0.90, 0.15, 0.05])
        button_grayscale = Button(but_grayscale, "Grayscale Image")
        button_grayscale.on_clicked(lambda event: self.grayscale_img())

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
        #Ensure f_transform exists
        if not hasattr(self, 'f_transform'):
            print("Perform Fourier Transform first!")
            return

        rows, cols = self.f_transform.shape
        center_row, center_col = rows // 2, cols // 2

        #Create LPF mask
        mask = np.zeros((rows, cols))
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - center_col)**2 + (y - center_row)**2 <= self.lpf**2
        mask[mask_area] = 1  # Retain low frequencies

        #Apply the mask
        self.filtered_transform = self.f_transform * mask

        #Update processed image to show the filtered Fourier transform
        filtered_magnitude = np.abs(self.filtered_transform)
        filtered_magnitude_log = np.log(filtered_magnitude + 1)
        filtered_magnitude_log = (filtered_magnitude_log - filtered_magnitude_log.min()) / (
            filtered_magnitude_log.max() - filtered_magnitude_log.min()
        )
        self.processed_img = filtered_magnitude_log
        self.update_display()


    def fft1d(self, x):
        """
        Function to apply the 1D fourier transform
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
        Function to apply the 1D inverse fourier transform
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
        Function to apply the 2D fourier transform
        """
        
        #Apply 1D FT on each row
        rows_ft = np.array([self.fft1d(row) for row in image])
        
        #Apply 1D FT on each column
        cols_ft = np.array([self.fft1d(col) for col in rows_ft.T]).T
        return cols_ft

    def i_fft2d(self, image_ft):
        """
        Function to apply the 2D inverse fourier transform
        """
        
        #Apply Inverse 1D IFT on each row
        rows_ift = np.array([self.i_fft1d(ft_row) for ft_row in image_ft])
        
        #Apply Inverse 1D IFT on each column
        cols_ifft = np.array([self.i_fft1d(ft_col) for ft_col in rows_ift.T]).T
        rows, cols = image_ft.shape

        #Return the real part for the image 
        return np.abs(cols_ifft) / (rows * cols)

    def shift_frequency(self, f_transform):
        """
        Function to make sure fourier transform frequencies are proplery alligned
        (low freq in the middle, etc.)
        """
        rows, cols = f_transform.shape

        #Swap quadrants to shift the zero-frequency component to center
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
        #Ensure filtered_transform exists
        if not hasattr(self, 'filtered_transform'):
            print("Apply LPF first")
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

    def sobel_gradiants(self, image):
        """
        Apply Sobel edge detection to the grayscale image.
        """
        if image.ndim != 2:
            print("Image must be grayscale for edge detection")
            return

        #Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        #Pad the image
        padded_image = np.pad(image, 1, mode='constant')
        rows, cols = image.shape

        #Initialize gradient matrices
        grad_x = np.zeros((rows, cols))
        grad_y = np.zeros((rows, cols))

        #Apply Sobel filters
        for i in range(rows):
            for j in range(cols):
                region = padded_image[i:i + 3, j:j + 3]
                grad_x[i, j] = np.sum(region * sobel_x)
                grad_y[i, j] = np.sum(region * sobel_y)
        
        return grad_x, grad_y


    def sobel_edge(self):
        """
        Function to apply sobel edge detection
        """
        grad_x, grad_y = self.sobel_gradiants(self.processed_img)

        #Compute gradient magnitude
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (grad_magnitude.max() - grad_magnitude.min())
        
        #Update processed image and display
        self.processed_img = grad_magnitude
        self.update_display()

    def gaussian_filter(self, image, kernel_size=5, sigma=1.4):
        """
        Apply Gaussian blur to reduce noise. Modified version of guassian filter
        from assignment 2, but takes no user input values, just default values
        """
        #Create Gaussian kernel
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        gauss = np.exp(-0.5 * (ax / sigma) ** 2)
        kernel = np.outer(gauss, gauss)
        kernel /= kernel.sum()

        #Convolve image with Gaussian kernel
        padded_image = np.pad(image, kernel_size // 2, mode='constant')
        rows, cols = image.shape
        blurred_img = np.zeros_like(image)

        for i in range(rows):
            for j in range(cols):
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                blurred_img[i, j] = np.sum(region * kernel)
        return blurred_img

    def non_maximum_suppression(self, gradient_magnitude, gradient_angle):
        """
        Apply Non-Maximum Suppression to thin edges
        """
        rows, cols = gradient_magnitude.shape
        nms = np.zeros((rows, cols), dtype=np.float64)
        angle = gradient_angle * 180.0 / np.pi
        angle[angle < 0] += 180

        #For each pixel compare direction of edge with left/right pixel accordingly
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                l, r = 255, 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    l = gradient_magnitude[i, j + 1]
                    r = gradient_magnitude[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    l = gradient_magnitude[i + 1, j - 1]
                    r = gradient_magnitude[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    l = gradient_magnitude[i + 1, j]
                    r = gradient_magnitude[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    l = gradient_magnitude[i - 1, j - 1]
                    r = gradient_magnitude[i + 1, j + 1]

                #If pixel value is greater than or equa to neighbours, it's local max, otherwise, set to 0
                if gradient_magnitude[i, j] >= l and gradient_magnitude[i, j] >= r:
                    nms[i, j] = gradient_magnitude[i, j]
                else:
                    nms[i, j] = 0
        return nms

    def edge_tracking(self, strong_edges, weak_edges):
        """
        Function to track edges
        """
        rows, cols = strong_edges.shape
        edges = np.copy(strong_edges)

        #For each pixel on weak edge, check neighbours, and if any are strong edges, promote
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if weak_edges[i, j] == 1:
                    if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                        edges[i, j] = 1
        return edges

    def canny_edge_detection(self):
        """
        Function to apply Canny Edge detection
        """
    
        #Apply gaussian blur
        blurred_img = self.gaussian_filter(self.processed_img, kernel_size=5, sigma=1.4)

        #Compute gradients
        grad_x, grad_y = self.sobel_gradiants(blurred_img)

        #Compute gradient magnitude and direction
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient_angle = np.arctan2(grad_y, grad_x)

        #Non Maximum suppression
        nms = self.non_maximum_suppression(gradient_magnitude, gradient_angle)

        #Double thresholding
        high_threshold = 0.2 * nms.max()
        low_threshold = 0.1 * high_threshold
        strong_edges = (nms >= high_threshold).astype(int)
        weak_edges = ((nms >= low_threshold) & (nms < high_threshold)).astype(int)

        #Edge Tracking 
        edges = self.edge_tracking(strong_edges, weak_edges)

        #Update and display the processed image
        self.processed_img = edges
        self.update_display()

if __name__ == "__main__":
    test = Assignment4()