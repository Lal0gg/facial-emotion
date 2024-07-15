import cv2
import os

# Define the path to the image
image_path = os.path.join("c:\\Users\\edujr\\OneDrive\\Documentos\\emotion\\facial-emotion", "angry.png")

# Check if the file exists
if not os.path.isfile(image_path):
    print(f"Error: File does not exist at path: {image_path}")
else:
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image from path: {image_path}")
    else:
        # Display the image
        cv2.imshow("Image", image)
        
        # Wait for the user to press a key
        cv2.waitKey(0)
        
        # Close all windows
        cv2.destroyAllWindows()
