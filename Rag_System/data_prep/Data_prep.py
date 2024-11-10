import fitz  # PyMuPDF
import re
from symspellpy import SymSpell, Verbosity

def extract_text_from_pdf(pdf_path, start_page, end_page):
    """
    Extract text from specified pages in a PDF file.
    """
    pdf = fitz.open(pdf_path)
    extracted_text = ""

    for page_num in range(start_page, end_page):
        page = pdf.load_page(page_num)  # Load each page
        extracted_text += page.get_text("text")  # Extract text from the page

    return extracted_text.replace("\n", "")

def clean_text(text):
    """
    Clean the extracted text by removing headers, footers, and specific patterns.
    """
    text = re.sub('[0-9]+.[0-9]+.[0-9]+.[0-9]+\t',' ', text)
    text = re.sub('[0-9]+.[0-9]+.[0-9]+\t',' ', text)
    text = re.sub('[0-9]+.[0-9]+\t',' ', text)
    text = re.sub('[[0-9]+.]+\t',' ', text)
    text = re.sub('[0-9]+.[0-9]+.[0-9]+.[0-9]+ ',' ', text)
    text = re.sub('[0-9]+.[0-9]+.[0-9]+ ',' ', text)
    text = re.sub('[0-9]+.[0-9]+ ',' ', text)
    text = re.sub('[[0-9]+.]+ ',' ', text)
    text = re.sub('[.[0-9]+]+\t',' ', text)
    text = re.sub('[.[0-9]+.]+ ',' ', text)
    text = re.sub('[0-9]+©2013 ',' ', text)
    text = re.sub("Project Management Institute.",' ', text)
    text = re.sub("Licensed To:.*?Reproduction.","", text)
    text = re.sub('\n',' ', text)
    text = re.sub(':',' conceptualise ', text)
    text = re.sub('†','', text)
    text = re.sub("™","'", text)
    text = re.sub("¥","'", text)
    text = re.sub("'s","'", text)
    text = re.sub("-",' ', text)
    text = re.sub("— ",'', text)
    text = re.sub('\"',' ', text)
    text = re.sub('“',' ', text)
    text = re.sub("[\(\[].*?[\)\]]", " ", text)
    text = re.sub("[ ]+"," ", text)
    text = re.sub("˜","fi", text)
    text = re.sub("˚","kn", text)
    text = re.sub("Š"," ", text)
    text = re.sub(","," ", text)
    text = re.sub("PMBOK®","PMBOK", text)
    text = re.sub("PROJECT RISK MANAGEMENTPROJECT RISK MANAGEMENT"," ", text)
    text = re.sub("Control RisksControl Risks"," ", text)
    text = re.sub("IDENTIFY RISKSIdentify Risks"," ", text)
    text = re.sub("PROJECT MANAGEMENT BODY OF KNOWLEDGE Œ "," ", text)
    text = re.sub("Monitor Risks"," ", text)
    text = re.sub("PRACTICE STANDARD FOR PROJECT RISK MANAGEMENT"," ", text)
    text = re.sub(r'[^.]*\binstitute\b[^.]*\.', '', text, flags=re.IGNORECASE)
    text = re.sub("[ ]+"," ", text)
    return text

def correct_spelling(text, dictionary_path):
    """
    Correct spelling in the text using SymSpell.
    """
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    
    corrected_text = ' '.join(
        sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)[0].term
        if sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        else word for word in text.split()
    )
    return corrected_text

def process_pdf(pdf_path, start_page, end_page, dictionary_path, output_file):
    """
    Full process of extracting, cleaning, and spell-correcting text from a PDF.
    """
    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_path, start_page, end_page)

    # Clean the text
    cleaned_text = clean_text(extracted_text)

    # Correct spelling
    corrected_text = correct_spelling(cleaned_text, dictionary_path)

    # Save the cleaned and corrected text to a file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(corrected_text)

    print(f"Processed text saved to {output_file}")
