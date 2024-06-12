![OMR-Evaluation](/OMR/image1.png)
# Automated OMR Sheet Grading System

## Project Description

The Automated Optical Mark Recognition (OMR) Sheet Grading System is designed to streamline the grading process of multiple-choice answer sheets using computer vision techniques. This system automates the detection and evaluation of marked answers on OMR sheets, calculates the percentage marks obtained by students, extracts roll numbers, and updates the results in an Excel sheet.

## Features

- **Automated Grading**: Automatically grades multiple-choice OMR sheets.
- **Roll Number Extraction**: Extracts student roll numbers from the OMR sheet.
- **Excel Update**: Updates student roll numbers and grades in an Excel sheet.

## Technologies Used

- Python
- OpenCV
- Flask
- Openpyxl
- Base64

## Prerequisites

- Python 3.x
- Pip (Python package installer)

## Installation

1. **Clone the Repository**:

    ```shell
   git clone https://github.com/your-username/OMR-Evaluation.git
   cd OMR-Evaluation
   ```
   
2. **Create a Virtual Environment** (optional but recommended):

    ```shell
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```    

3. **Install Dependencies**:

    ```shell
    pip install -r requirements.txt
   ```
   
## Usage

1. **Run the Flask App**:

    ```sh
   cd OMR
    python check.py
    ```

2. **Access the Web Interface**:
   
    Open your web browser and navigate to `http://127.0.0.1:5000`.

3. **Upload an Image**:
   
    - Click on the "Upload Image" button.
    - Select the OMR sheet image file from your local system.
    - Click the "Process Image" button to upload and process the image.

4. **View Results**:
   
    - The system will process the uploaded image, extract the roll number and marks, and update the Excel sheet.
    - The results will be displayed on the webpage and stored in the specified Excel file `students.xlsx`.
