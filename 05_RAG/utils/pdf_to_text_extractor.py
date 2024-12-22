import os
from PyPDF2 import PdfReader

def extract_text_from_pdfs(pdf_folder, output_file):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create or open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as txt_file:
        # Iterate over all files in the specified folder
        for filename in os.listdir(pdf_folder):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(pdf_folder, filename)
                # Open the PDF file
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_reader = PdfReader(pdf_file)
                    # Extract text from each page
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            txt_file.write(text)
                            txt_file.write('\n')  # Add a newline after each page's text

if __name__ == "__main__":
    # Get the current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the input folder and output folder paths
    input_folder_path = os.path.join(current_dir, '../google_whitepaper_rag_content')
    output_folder_path = os.path.join(current_dir, '../content_for_rag')
    output_file_path = os.path.join(output_folder_path, 'google_ai_engineer_notes.txt')
    
    # Extract text from PDFs and store the result
    extract_text_from_pdfs(input_folder_path, output_file_path)