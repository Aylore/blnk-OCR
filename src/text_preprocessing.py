import re
import unicodedata
import cv2

class TextProcessing:
    def __init__(self):
        pass

   
    def remove_english_letters(self , text):
        # Define a regular expression pattern to match English letters
        pattern = r'[a-zA-Z]'

        # Use re.sub() to remove English letters from the text
        cleaned_text = re.sub(pattern, '', text)

        return cleaned_text

   
    def remove_arabic_diacritics(self ,text):
        # Remove Arabic diacritics using Unicode normalization
        normalized_text = unicodedata.normalize('NFD', text)
        cleaned_text = ''.join([c for c in normalized_text if not unicodedata.combining(c)])
        
        return cleaned_text

    @staticmethod
    def extract_black_text(img_path, save=False):
        # Load your image
        image = cv2.imread(img_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert the binary image (black text on white background)
        inverted_image = cv2.bitwise_not(binary_image)

        if save:
            # Save the extracted black text as a separate image
            cv2.imwrite('black_text_extracted.jpg', inverted_image)
        
        return inverted_image
    

    def apply_all(self ,text):
        text = self.remove_arabic_diacritics(text)
        text = self.remove_english_letters(text)

        return text
    

if __name__ =="main":
    # Example usage:
    text_processor = TextProcessing()
    cleaned_text = text_processor.remove_english_letters("Hello World")
    cleaned_arabic_text = text_processor.remove_arabic_diacritics("مرحبًا بكم")
    extracted_image = text_processor.extract_black_text("your_image.jpg", save=True)
