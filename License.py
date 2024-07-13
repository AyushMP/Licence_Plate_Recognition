import cv2
import pytesseract

# Set up pytesseract path (customize the path if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the Haar Cascade for license plate detection
cascade_path = "haarcascade_russian_plate_number.xml"
license_plate_cascade = cv2.CascadeClassifier(cascade_path)

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def detect_license_plate(image, cascade):
    # Detect license plates
    plates = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    return plates

def recognize_characters(license_plate_image):
    # Convert to grayscale (if not already)
    gray_license_plate = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    _, thresh_license_plate = cv2.threshold(gray_license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # OCR recognition with custom configuration
    config = r'--oem 3 --psm 7 -l eng'  # Adjust language if needed
    text = pytesseract.image_to_string(thresh_license_plate, config=config)
    return text

def main(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Preprocess the image
    gray_image = preprocess_image(image)
    # Detect license plates
    plates = detect_license_plate(gray_image, license_plate_cascade)

    if len(plates) > 0:
        for (x, y, w, h) in plates:
            # Extract license plate
            license_plate_image = image[y:y + h, x:x + w]
            # Recognize characters on the license plate
            license_plate_text = recognize_characters(license_plate_image)
            print(f"License Plate Text: {license_plate_text}")
            # Draw rectangle around the detected license plate and display the result
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("License Plate Detection", image)
            cv2.imshow("License Plate", license_plate_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No license plate detected.")

if __name__ == "__main__":
    main(r"images\gettyimages-963767120_wide-a7ce9125834d9599dc4609847ce9b688fe6e7d28.jpg")  # Use a raw string (prefix with 'r') to handle backslashes in the path
