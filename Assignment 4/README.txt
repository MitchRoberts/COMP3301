In order to run the assignment, please click on Assignment4.py, and click run. You will then be prompted to select an image. Once you select the image,
The GUI will pop up where you are able to Convert image to grayscale, and more. You can then click the reset image button
in the top right to reset the image and continue working with it. 

In order to apply Canny or Sobel, you must first hit the grayscale image button on the top
middle of the screen, otherwise it won't let you.

The fourier transform button automatically converts image to grayscale, although you are still able to change to grayscale first and then click the button
and compute the FFT of the image, using 1D FFT functions. Once you compute the FT of the image, you are then able to apply the LPF filter by choosing a value
in the top left, then clicking apply filter. This will then show the filtered image on the right, and then you are able to click the inverse FT button to show the 
resulting image.

The libraries that were used are:
    PyQT/sys: File exploer for loading image
    MatPlotLib: Only for GUI, no built in functions were used apart from creating GUI
    Numpy: Padding, array conversions/operations
    PIL: Loading images

Added Feature: Reset Image button

