import cv2
import pytesseract
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

class BookRecognizer:
    def __init__(self, images_dir):
        # Initialize Google Books API endpoint
        self.books_api_url = "https://www.googleapis.com/books/v1/volumes"
        self.images_dir = Path(images_dir)
        
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('GOOGLE_BOOKS_API_KEY')
        
        # Supported image extensions
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    def get_image_files(self):
        """Get all image files from the specified directory"""
        return [f for f in self.images_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in self.image_extensions]

    def process_image(self, image_path):
        """Process the image and extract text"""
        # Read the image
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Extract text from the image
        text = pytesseract.image_to_string(gray)
        
        return text.strip()

    def search_book(self, query):
        """Search for book information using Google Books API"""
        params = {
            'q': query,
            'key': self.api_key
        }
        
        try:
            response = requests.get(self.books_api_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'items' in data and len(data['items']) > 0:
                book = data['items'][0]['volumeInfo']
                return {
                    'title': book.get('title', 'Unknown'),
                    'authors': book.get('authors', ['Unknown']),
                    'publisher': book.get('publisher', 'Unknown'),
                    'published_date': book.get('publishedDate', 'Unknown'),
                    'description': book.get('description', 'No description available'),
                    'isbn': book.get('industryIdentifiers', [{'identifier': 'Unknown'}])[0]['identifier']
                }
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error searching for book: {e}")
            return None

    def run(self):
        """Main function to process all images in the directory"""
        image_files = self.get_image_files()
        
        if not image_files:
            print(f"No image files found in {self.images_dir}")
            return

        print(f"Found {len(image_files)} images to process")
        results = []

        for image_path in image_files:
            print(f"\nProcessing {image_path.name}...")
            
            # Extract text from image
            extracted_text = self.process_image(image_path)
            
            if extracted_text:
                print("Searching for book information...")
                # Search for book information
                book_info = self.search_book(extracted_text)
                
                if book_info:
                    results.append({
                        'image': image_path.name,
                        'info': book_info
                    })
                    print(f"✓ Found: {book_info['title']} by {', '.join(book_info['authors'])}")
                else:
                    print(f"✗ No book information found for {image_path.name}")
            else:
                print(f"✗ No text could be extracted from {image_path.name}")

        # Print final summary
        print("\n=== Summary ===")
        print(f"Successfully processed {len(results)} out of {len(image_files)} images\n")
        
        # Save results to a file
        self.save_results(results)

    def save_results(self, results):
        """Save the results to a text file"""
        output_file = self.images_dir / 'book_results.txt'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"\nImage: {result['image']}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Title: {result['info']['title']}\n")
                f.write(f"Authors: {', '.join(result['info']['authors'])}\n")
                f.write(f"Publisher: {result['info']['publisher']}\n")
                f.write(f"Published Date: {result['info']['published_date']}\n")
                f.write(f"ISBN: {result['info']['isbn']}\n")
                f.write("\nDescription:\n")
                f.write(f"{result['info']['description']}\n")
                f.write("=" * 50 + "\n")

        print(f"Results have been saved to {output_file}")

if __name__ == "__main__":
    # Replace this with your images directory path
    IMAGES_DIR = "book_images"
    
    if not os.path.exists(IMAGES_DIR):
        print(f"Creating directory: {IMAGES_DIR}")
        os.makedirs(IMAGES_DIR)
        print(f"Please place your book cover images in the '{IMAGES_DIR}' directory")
    else:
        recognizer = BookRecognizer(IMAGES_DIR)
        recognizer.run() 