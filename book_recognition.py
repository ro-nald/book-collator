import cv2
import pytesseract
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import time
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from skimage.feature import ORB
from skimage.color import rgb2gray

class RateLimiter:
    def __init__(self, calls_per_minute=60):
        self.calls_per_minute = calls_per_minute
        self.calls = deque()

    def wait_if_needed(self):
        """Wait if we've exceeded our rate limit"""
        now = datetime.now()

        # Remove calls older than 1 minute
        while self.calls and self.calls[0] < now - timedelta(minutes=1):
            self.calls.popleft()

        # If we've made too many calls in the last minute, wait
        if len(self.calls) >= self.calls_per_minute:
            # Calculate how long to wait
            oldest_call = self.calls[0]
            wait_time = (oldest_call + timedelta(minutes=1) - now).total_seconds()
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

        # Add this call to the queue
        self.calls.append(now)

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

        # Initialize rate limiter (Google Books API default is 1000 queries per day)
        # Setting a conservative limit of 60 calls per minute
        self.rate_limiter = RateLimiter(calls_per_minute=60)

        # Initialize ORB feature detector for cover similarity
        self.feature_detector = ORB(n_keypoints=1000)

        # Dictionary to store feature descriptors for all images
        self.image_features = {}

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
        # Wait if we need to due to rate limiting
        self.rate_limiter.wait_if_needed()

        params = {
            'q': query,
            'key': self.api_key
        }

        try:
            response = requests.get(self.books_api_url, params=params)
            response.raise_for_status()

            # Check for rate limit response codes
            if response.status_code == 429:  # Too Many Requests
                print("Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)  # Wait a minute
                return self.search_book(query)  # Retry the request

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
            # If we get a rate limit error, wait and retry
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                print("Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)  # Wait a minute
                return self.search_book(query)  # Retry the request
            return None

    def extract_features(self, image_path):
        """Extract ORB features from an image"""
        # Read and preprocess image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None

        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to grayscale
        if len(image.shape) == 3:
            image = rgb2gray(image)

        self.feature_detector.detect_and_extract(image)
        return self.feature_detector.keypoints, self.feature_detector.descriptors

    def compute_similarity(self, desc1, desc2):
        """Compute similarity score between two feature descriptors"""
        if desc1 is None or desc2 is None:
            return 0

        # Use Hamming distance for binary descriptors
        matches = 0
        for d1 in desc1:
            # Find the closest descriptor in desc2
            distances = np.sum(d1 != desc2, axis=1)
            min_dist = np.min(distances)
            if min_dist < 32:  # threshold for good match
                matches += 1

        # Normalize similarity score
        similarity = matches / min(len(desc1), len(desc2))
        return similarity

    def group_similar_covers(self, threshold=0.3):
        """Group similar book covers together"""
        image_files = self.get_image_files()

        if not image_files:
            print(f"No image files found in {self.images_dir}")
            return []

        print(f"Found {len(image_files)} images to process")
        print("\nAnalyzing cover similarities...")

        # Extract features for all images
        for image_path in image_files:
            if str(image_path) not in self.image_features:
                _, descriptors = self.extract_features(image_path)
                self.image_features[str(image_path)] = descriptors

        # Create groups of similar images
        groups = []
        processed = set()

        for image_path in image_files:
            if str(image_path) in processed:
                continue

            current_group = [image_path]
            processed.add(str(image_path))

            # Compare with all other unprocessed images
            for other_path in image_files:
                if str(other_path) in processed:
                    continue

                similarity = self.compute_similarity(
                    self.image_features[str(image_path)],
                    self.image_features[str(other_path)]
                )

                if similarity > threshold:
                    current_group.append(other_path)
                    processed.add(str(other_path))

            groups.append(current_group)

        print(f"Found {len(groups)} distinct cover groups")
        return groups

    def run(self):
        """Main function to process all images in the directory"""
        # First, group similar covers
        groups = self.group_similar_covers()
        if not groups:
            return

        results = []

        # Process each group
        for group_idx, group in enumerate(groups, 1):
            print(f"\nProcessing group {group_idx} ({len(group)} images)...")

            # Use the first image in the group as the representative
            representative_image = group[0]
            print(f"Using {representative_image.name} as representative image")

            # Extract text from representative image
            extracted_text = self.process_image(representative_image)

            if extracted_text:
                print("Searching for book information...")
                # Search for book information
                book_info = self.search_book(extracted_text)

                if book_info:
                    group_result = {
                        'representative_image': representative_image.name,
                        'similar_images': [img.name for img in group[1:]],
                        'info': book_info
                    }
                    results.append(group_result)

                    print(f"✓ Found: {book_info['title']} by {', '.join(book_info['authors'])}")
                    if len(group) > 1:
                        print("Similar covers in this group:")
                        for img in group[1:]:
                            print(f"  - {img.name}")
                else:
                    print(f"✗ No book information found for {representative_image.name}")
            else:
                print(f"✗ No text could be extracted from {representative_image.name}")

        # Print final summary
        print("\n=== Summary ===")
        print(f"Successfully processed {len(results)} out of {len(groups)} groups\n")

        # Save results to a file
        self.save_results(results)

    def save_results(self, results):
        """Save the results to a text file"""
        output_file = self.images_dir / 'book_results.txt'

        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"\nGroup Representative Image: {result['representative_image']}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Title: {result['info']['title']}\n")
                f.write(f"Authors: {', '.join(result['info']['authors'])}\n")
                f.write(f"Publisher: {result['info']['publisher']}\n")
                f.write(f"Published Date: {result['info']['published_date']}\n")
                f.write(f"ISBN: {result['info']['isbn']}\n")
                f.write("\nDescription:\n")
                f.write(f"{result['info']['description']}\n")

                if result['similar_images']:
                    f.write("\nSimilar Covers in Group:\n")
                    for img in result['similar_images']:
                        f.write(f"  - {img}\n")

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