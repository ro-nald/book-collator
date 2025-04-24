# Book Recognition App

This application uses OpenCV and Tesseract OCR to recognize books from their cover images and retrieve detailed information using the Google Books API.

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
- Process all images in the `book_images` directory
- Extract text from each image
- Search for book information using the Google Books API
- Save the results to `book_results.txt` in the `book_images` directory

## Tips for Best Results

1. Use clear, well-lit images of book covers
2. Ensure the text on the covers is readable
3. Images should be right-side up (not rotated)
4. Avoid glare or reflections on the book covers

## Output

The results will be saved in `book_images/book_results.txt` and will include:
- Book title
- Authors
- Publisher
- Publication date
- ISBN
- Description 