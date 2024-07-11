import cv2
import pytesseract
import numpy as np

# Set up pytesseract path (customize the path if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection
    edged = cv2.Canny(blurred, 50, 200)
    return edged

def detect_license_plate(edged_image):
    # Find contours in the edged image
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    license_plate_contour = None
    max_area = 0

    for contour in contours:
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 10, True)
        area = cv2.contourArea(contour)
        # Filter contours by size and shape
        if len(approx) == 4 and area > max_area:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 5:  # Aspect ratio check
                license_plate_contour = approx
                max_area = area

    return license_plate_contour

def extract_license_plate(image, contour):
    if contour is None:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    license_plate_image = image[y:y + h, x:x + w]
    return license_plate_image

def recognize_characters(license_plate_image):
    # Convert to grayscale (if not already)
    gray_license_plate = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    thresh_license_plate = cv2.adaptiveThreshold(gray_license_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # OCR recognition with custom configuration
    config = '--oem 3 --psm 7'  # Optimal settings for single text line
    text = pytesseract.image_to_string(thresh_license_plate, config=config)
    return text

def main(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Preprocess the image
    edged_image = preprocess_image(image)
    # Detect license plate
    license_plate_contour = detect_license_plate(edged_image)
    # Extract license plate
    license_plate_image = extract_license_plate(image, license_plate_contour)

    if license_plate_image is not None:
        # Recognize characters on the license plate
        license_plate_text = recognize_characters(license_plate_image)
        print(f"License Plate Text: {license_plate_text}")
        # Draw contour and display the result
        cv2.drawContours(image, [license_plate_contour], -1, (0, 255, 0), 3)
        cv2.imshow("License Plate Detection", image)
        cv2.imshow("License Plate", license_plate_image)
        cv2.imshow("Thresholded License Plate", cv2.cvtColor(cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No license plate detected.")

if __name__ == "__main__":
    main("images (1).jpeg")
