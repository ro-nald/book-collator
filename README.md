# Book Recognition App

This application uses OpenCV and Tesseract OCR to recognize books from their cover images, group similar covers together, and retrieve detailed information using the Google Books API.

## Prerequisites

1. Python 3.7 or higher
2. Tesseract OCR installed on your system
   - For macOS: `brew install tesseract`
   - For Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - For Windows: Download and install from [here](https://github.com/UB-Mannheim/tesseract/wiki)

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Get a Google Books API key:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Books API
   - Create credentials (API key)
   - Copy the API key

4. Create your `.env` file:
   - Copy `.env.example` to `.env`
   - Replace `your_api_key_here` with your actual Google Books API key

## Usage

1. Create a directory called `book_images` in the project root (if it doesn't exist already)
2. Place your book cover images in the `book_images` directory
   - Supported formats: .jpg, .jpeg, .png, .bmp, .tiff
3. Run the application:
```bash
python book_recognition.py
```

The application will:
1. Analyze all book covers and group similar ones together
2. Process one representative image from each group
3. Extract text from the representative image
4. Search for book information using the Google Books API (with rate limiting)
5. Save the results to `book_results.txt` in the `book_images` directory

## Features

### Cover Similarity Detection
- Uses ORB (Oriented FAST and Rotated BRIEF) features to detect similar book covers
- Groups similar covers together to avoid duplicate processing
- Adjustable similarity threshold (default: 0.3)

### Rate Limiting
- Implements intelligent rate limiting for Google Books API calls
- Default: 60 calls per minute (configurable)
- Automatic retry with backoff for rate limit errors

## Tips for Best Results

1. Use clear, well-lit images of book covers
2. Ensure the text on the covers is readable
3. Images should be right-side up (not rotated)
4. Avoid glare or reflections on the book covers
5. For similar cover detection:
   - Use consistent lighting across images
   - Try to maintain similar angles when photographing covers
   - Adjust similarity threshold if needed (in code)

## Output

The results will be saved in `book_images/book_results.txt` and will include:
- Representative image for each group
- List of similar covers found
- Book information:
  - Title
  - Authors
  - Publisher
  - Publication date
  - ISBN
  - Description

## Performance Notes

- The script processes images in groups to minimize API calls
- Similar covers are detected using computer vision techniques before making API calls
- Rate limiting ensures reliable API access without hitting quota limits 