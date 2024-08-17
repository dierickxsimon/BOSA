import cv2 as cv
import numpy as np

def noop(x):
    pass

def create_mask(image, min_hsv, max_hsv):
    # Convert image to HSV color space
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Create mask using the provided HSV range
    mask = cv.inRange(image_hsv, min_hsv, max_hsv)
    
    return mask

def main():
    # Create a window to set mask parameters
    SET_MASK_WINDOW = "Set Mask"
    cv.namedWindow(SET_MASK_WINDOW, cv.WINDOW_NORMAL)
    
    # Create trackbars for HSV values
    cv.createTrackbar("Min Hue", SET_MASK_WINDOW, 0, 179, noop)
    cv.createTrackbar("Max Hue", SET_MASK_WINDOW, 179, 179, noop)
    cv.createTrackbar("Min Sat", SET_MASK_WINDOW, 0, 255, noop)
    cv.createTrackbar("Max Sat", SET_MASK_WINDOW, 255, 255, noop)
    cv.createTrackbar("Min Val", SET_MASK_WINDOW, 0, 255, noop)
    cv.createTrackbar("Max Val", SET_MASK_WINDOW, 255, 255, noop)

    # Read input image
    image = cv.imread('renners.png')

    while True:
        # Get current HSV values from trackbars
        min_hue = cv.getTrackbarPos("Min Hue", SET_MASK_WINDOW)
        max_hue = cv.getTrackbarPos("Max Hue", SET_MASK_WINDOW)
        min_sat = cv.getTrackbarPos("Min Sat", SET_MASK_WINDOW)
        max_sat = cv.getTrackbarPos("Max Sat", SET_MASK_WINDOW)
        min_val = cv.getTrackbarPos("Min Val", SET_MASK_WINDOW)
        max_val = cv.getTrackbarPos("Max Val", SET_MASK_WINDOW)

        # Create minimum and maximum HSV arrays
        min_hsv = np.array([min_hue, min_sat, min_val])
        max_hsv = np.array([max_hue, max_sat, max_val])

        # Create mask
        mask = create_mask(image, min_hsv, max_hsv)

        # Apply mask to input image
        result_image = cv.bitwise_and(image, image, mask=mask)

        # Display images
        cv.imshow("Input Image", image)
        cv.imshow("Result Image", result_image)

        # Break the loop if Esc key is pressed
        if cv.waitKey(1) == 27:
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
