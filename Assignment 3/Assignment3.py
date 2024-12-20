import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from PIL import Image
from PyQt5.QtWidgets import QApplication, QFileDialog
import math

class Image_Thresholding:

    def __init__(self):
        #Init a variable for storing future istogram data
        self.histograms = None

        #Prompt the user to select an image file
        self.image_array = self.load_image_via_dialog()

        #Copy image for future processing
        self.processed_img = self.image_array.copy()

        #Setup figures and axes
        self.fig, (self.axes1, self.axes2, self.axes3) = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [0.5, 2, 0.5]})

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

        #Initalize values
        self.man_thresh = 128
        self.offset = 10

        self.histograms = self.display_histogram(self.processed_img)

        if self.processed_img.ndim == 3:
            colours = ['red', 'green', 'blue']
            for i in range(3):
                mean_intensity = np.mean(self.processed_img[:, :, i]) * 255
                self.axes2.axvline(x=mean_intensity, color=colours[i], linestyle='--')
                print(f"Mean intensity for {colours[i]}: {mean_intensity}")
        elif self.processed_img.ndim == 2:
            mean_intensity = np.mean(self.processed_img) * 255
            self.axes2.axvline(x=mean_intensity, color='black', linestyle='--')

           


    # Create buttons and a text box
        self.setup_buttons()

    # Set up layout and show the plot
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

        #Manual Threshold button
        but_manual_thresh = plt.axes([0.05, 0.01, 0.15, 0.05])
        button_manual_thresh = Button(but_manual_thresh, "Manual Threshold")
        button_manual_thresh.on_clicked(lambda event: self.manual_threshold())

        #Automatic Threshold button
        but_auto_thresh = plt.axes([0.25, 0.01, 0.15, 0.05])
        button_auto_thresh = Button(but_auto_thresh, "Auto Threshold")
        button_auto_thresh.on_clicked(lambda event: self.auto_threshold())

        #Otsu's Threshold button
        but_otsu_thresh = plt.axes([0.45, 0.01, 0.15, 0.05])
        button_otsu_thresh = Button(but_otsu_thresh, "Otsu Threshold")
        button_otsu_thresh.on_clicked(lambda event: self.otsu_threshold())

        #Adaptive Threshold button
        but_adaptive_thresh = plt.axes([0.65, 0.01, 0.20, 0.05])
        button_adaptive_thresh = Button(but_adaptive_thresh, "Adaptive Threshold")
        button_adaptive_thresh.on_clicked(lambda event: self.adaptive_threshold())

        #Threshold value text input for Manual Thresholding
        thresh_box_txt = plt.axes([0.05, 0.90, 0.1, 0.05])
        thresh_box = TextBox(thresh_box_txt, 'Threshold', initial="128")
        thresh_box.on_submit(self.update_threshold)

        #Offset value text input for Adaptive Thresholding
        offset_box_txt = plt.axes([0.20, 0.90, 0.1, 0.05])
        offset_box = TextBox(offset_box_txt, 'Offset', initial="10")
        offset_box.on_submit(self.update_offset)

        #Reset image button
        but_reset = plt.axes([0.75, 0.90, 0.20, 0.05])
        button_reset = Button(but_reset, "Reset Image")
        button_reset.on_clicked(lambda event: self.reset_image())

        plt.show()

    def update_threshold(self, text):
        """
        Update manual threshold value from user input
        """
        self.man_thresh = float(text)
        print(f"Manual Threshold Value updated to {self.man_thresh}")
        self.axes2.clear()
        self.histograms = self.display_histogram(self.processed_img)
        if self.processed_img.ndim == 2:
            self.axes2.axvline(x=self.man_thresh, color='black', linestyle='--')

        elif self.processed_img.ndim == 3:
            colours = ['red', 'green', 'blue']
            for i in range(3):
                self.axes2.axvline(x=self.man_thresh, color=colours[i], linestyle='--')

        
        return self.man_thresh
    
    def update_offset(self, text):
        """
        Update offset value from user input
        """
        self.offset = float(text)
        print(f"Offset Value updated to {self.offset}")
        return self.offset

    def reset_image(self):
        """
        Function for resetting to original image
        """
        print("Reseting to original image...")
        self.processed_img = self.image_array.copy()
        self.axes2.clear()

        #Update display
        if self.processed_img.ndim == 2:
            self.axes3.imshow(self.processed_img, cmap='gray')
        else:
            self.axes3.imshow(self.processed_img)
        
        self.histograms = self.display_histogram(self.processed_img)
        plt.draw() 

    def display_histogram(self, image, display=True):
        """
        Function used for displaying histograms and returning data
        If we just want to return data, we can set display to false
        """
        #self.axes2.clear()

        histogram = []

        #Check if grayscale image
        if image.ndim == 2:
            histogram = [0] * 256

            #For each pixel in each row, check intensity values and increment
            #value in the histogram array if intensity value matches
            for row in image:
                for pixel in row:
                    intensity_val = int(pixel * 255)
                    histogram[intensity_val] += 1

            if display == True:
                #Plot bins for histogram, and draw vertical line for threshold
                self.axes2.plot(range(256), histogram, color='black')

        #Check if RGB image
        elif image.ndim == 3:
            histogram_r = [0] * 256
            histogram_g = [0] * 256 
            histogram_b = [0] * 256

            #For each pixel in each row, for all channels, check intensity values 
            #and increment value in each channel of the histogram array if intensity 
            #values matches
            for row in image:
                for pixel in row:
                    r, g, b = (int(pixel[0] * 255), int(pixel[1] * 255), int(pixel[2] * 255))
                    histogram_r[r] += 1
                    histogram_g[g] += 1
                    histogram_b[b] += 1

            if display == True:
                #Plot lines for histogram and store in histograms variable
                self.axes2.plot(range(256), histogram_r, color='red')
                self.axes2.plot(range(256), histogram_g, color='green')
                self.axes2.plot(range(256), histogram_b, color='blue')
            histogram = [histogram_r, histogram_g, histogram_b]

        if display == True:
            #Update display and set axes labels
            self.axes2.set_xlabel('Intensity')
            self.axes2.set_ylabel('Frequency')
            self.axes2.set_title('Image Histogram')
            self.fig.canvas.draw_idle()

        return histogram

    def manual_threshold(self):
        """
        Apply the manual thresholding function
        """
        self.axes2.clear()
        print("Applying Manual Threshold...")
        threshold = self.man_thresh / 255
        

        #Update display
        if self.processed_img.ndim == 2:
            self.processed_img = np.where(self.processed_img >= (threshold), 1, 0)
            self.axes2.axvline(x=threshold * 256, color='black', linestyle='--')
            self.axes3.imshow(self.processed_img, cmap='gray')
        else:
            colours = ['red', 'green', 'blue']
            for i in range(3):
                self.processed_img[:, :, i] = np.where(self.processed_img[:, :, i] >= threshold, 1, 0)
                self.axes2.axvline(x=threshold * 256, color=colours[i], linestyle='--')
            self.axes3.imshow(self.processed_img)

        self.histograms = self.display_histogram(self.processed_img)
        plt.draw()

    def auto_threshold(self):
        """
        Apply the automatic thresholding function according to the image histogram
        """
        self.axes2.clear()

        #Define number that will act as guard to keep algo going untill we go below
        min_num = 0.0005

        if self.processed_img.ndim == 2:
            #and deltaT as large number
            deltaT = 1000000000
            #Init T(initial threshold) as mean of image
            T = np.mean(self.processed_img)

            #While delta T is greater than the pre defined min number, split the histogram into 2 groups,
            #Calculate the mean of each group, find average, and assign the abs(T - Tnew) to delta T, then assign T as
            #the new T
            while deltaT > min_num:
                G1 = self.processed_img[self.processed_img <= T]
                G2 = self.processed_img[self.processed_img > T]

                #Check case for infinite loop
                if len(G1) == 0 or len(G2) == 0:
                    break

                mew1 = np.mean(G1)
                mew2 = np.mean(G2)

                Tnew = (mew1 + mew2) / 2

                deltaT = abs(Tnew - T)

                T = Tnew

            final_threshold = T

            self.processed_img = np.where(self.processed_img >= final_threshold, 1, 0)

        elif self.processed_img.ndim == 3:

            #While delta T is greater than the pre defined min number, split the histogram into 2 groups,
            #Calculate the mean of each group, find average, and assign the abs(T - Tnew) to delta T, then assign T as
            #the new T, now repeat for each colour channel
            for i in range(3):
                thresholds = []
                #and deltaT as large number
                deltaT = 1000000000
                T = np.mean(self.processed_img[:, :, i])
                while deltaT > min_num:

                    G1 = self.processed_img[:, :, i][self.processed_img[:, :, i] <= T]
                    G2 = self.processed_img[:, :, i][self.processed_img[:, :, i] > T]

                    #Check case for infinite loop
                    if len(G1) == 0 or len(G2) == 0:
                        break

                    mew1 = np.mean(G1)
                    mew2 = np.mean(G2)

                    Tnew = (mew1 + mew2) / 2

                    deltaT = abs(T - Tnew)

                    T = Tnew

                final_threshold = T
                thresholds.append(T)

                self.processed_img[:, :, i] = np.where(self.processed_img[:, :, i] >= final_threshold, 1, 0)

        #Update display
        if self.processed_img.ndim == 2:
            self.axes3.imshow(self.processed_img, cmap='gray')
        else:
            self.axes3.imshow(self.processed_img)
           

        self.histograms = self.display_histogram(self.processed_img)
        plt.draw()
        

    def otsu_threshold(self):
        """
        Applys otsu thresholding method
        """
        self.axes2.clear()

        thresholds = []
        #Check if RGB
        if self.processed_img.ndim == 3:
            self.histograms = self.display_histogram(self.processed_img, display=False)

            #For each color channel get probability distribution of intensity levels
            for i in range(3):
                histogram = np.array(self.histograms[i])
                pixels = histogram.sum()

                probability = histogram / pixels

                #Dot product of intensity levels and probability distribution
                mean_total = np.dot(np.arange(256), probability)

                #Initialize variables
                max = 0
                optimal_threshold = 0
                w0 = 0  #Probability of class 0
                sum0 = 0  #Sum for class 0

                #Iterate through all possible ranges of threshold values, add to w0
                for j in range(256):
                    w0 += probability[j]

                    #Avoid division by 0
                    if w0 == 0:
                        continue

                    #Probability of class 1
                    w1 = 1 - w0
                    if w1 == 0:
                        break

                    #Update class 0 sum and calc the means for both classes
                    sum0 += j * probability[j]
                    mean0 = sum0 / w0
                    mean1 = (mean_total - sum0) / w1

                    #Compute variance
                    variance = w0 * w1 * (mean0 - mean1) ** 2

                    #Update variance and optimal threshold accordingly
                    if variance > max:
                        max = variance
                        optimal_threshold = j

                #Store optimal threshold for the corresponding channel
                thresholds.append(optimal_threshold)

            #Apply optimal threshold for each channel
            for i in range(3):
                self.processed_img[:, :, i] = np.where(self.processed_img[:, :, i] >= thresholds[i] / 255, 1, 0)

        #Check if grayscale
        elif self.processed_img.ndim == 2:
            histogram = np.array(self.display_histogram(self.processed_img, display=False))
            pixels = histogram.sum()

            #Get probability distribution of each intensity level
            probability = histogram / pixels

            #Dot product of intensity levels and probability distribution
            mean_total = np.dot(np.arange(256), probability)

            #Initialize variables
            max = 0
            optimal_threshold = 0
            w0 = 0  #Probability of class 0
            sum0 = 0  #Sum of class 0

            #Iterate through all possible ranges of threshold values, add to w0
            for j in range(256):
                w0 += probability[j]

                #Avoid division by 0
                if w0 == 0:
                    continue

                #Probability of class 1
                w1 = 1 - w0
                if w1 == 0:
                    break

                #Update class 0 sum and calc the means for both classes
                sum0 += j * probability[j]
                mean0 = sum0 / w0
                mean1 = (mean_total - sum0) / w1

                #Compute variance
                variance = w0 * w1 * (mean0 - mean1) ** 2

                #Update variance and optimal threshold accordingly
                if variance > max:
                    max = variance
                    optimal_threshold = j

            #Apply optimal threshold
            self.processed_img = np.where(self.processed_img >= optimal_threshold / 255, 1, 0)
        
        #Update display
        if self.processed_img.ndim == 2:
            self.axes2.axvline(x=optimal_threshold, color='black', linestyle='--')
            self.axes3.imshow(self.processed_img, cmap='gray')
        else:
            self.axes3.imshow(self.processed_img)
            colours = ['red', 'green', 'blue']
            for i in range(3):
                self.axes2.axvline(x=thresholds[i], color=colours[i], linestyle='--')

        self.histograms = self.display_histogram(self.processed_img)
        plt.draw()

    def adaptive_threshold(self):
        """
        Applys adaptive thresholding method (Mean C approach)
        """
        self.axes2.clear()

        #Initialize window size and offset
        window_size = 7
        offset = self.offset / 255

        temp_img = self.processed_img.copy()

        #Check if grayscale
        if self.processed_img.ndim == 2:

            #Pad image for border handling
            img_pad = np.pad(self.processed_img, window_size // 2, mode='constant', constant_values=0)

            #Iterate over each pixel of image
            for i in range(self.processed_img.shape[0]):
                for j in range(self.processed_img.shape[1]):

                    #Get window around pixel, calculate the mean and apply the threshold
                    #with user inputted offset value
                    window = img_pad[i:i + window_size, j:j + window_size]
                    mean = np.mean(window)
                    if self.processed_img[i,j] >= (mean - offset):
                        temp_img[i,j] = 1
                    else:
                        temp_img[i,j] = 0
        
        #Check if RGB
        if self.processed_img.ndim  == 3:

            #Iterate over each channel
            for channel in range(3):

                #Pad channel for border handling
                img_pad_channel = np.pad(self.processed_img[:, :, channel], window_size // 2, mode='constant', constant_values=0)

                #Iterate over each pixel of the channel
                for i in range(self.processed_img.shape[0]):
                    for j in range(self.processed_img.shape[1]):

                        window = img_pad_channel[i:i + window_size, j:j + window_size]
                        mean = np.mean(window)
                        if self.processed_img[i, j, channel] >= (mean - offset):
                            temp_img[i, j, channel] = 1
                        else:
                            temp_img[i, j, channel] = 0
        
        #Assign image to temp image
        self.processed_img = temp_img

         #Update display
        if self.processed_img.ndim == 2:
            self.axes3.imshow(self.processed_img, cmap='gray')
        else:
            self.axes3.imshow(self.processed_img)

        self.histograms = self.display_histogram(self.processed_img)
        plt.draw()


if __name__ == "__main__":
    test = Image_Thresholding()