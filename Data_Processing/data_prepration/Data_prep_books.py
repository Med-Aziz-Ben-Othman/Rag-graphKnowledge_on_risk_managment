# text_processing_pipeline.py

import fitz
import re
import string
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from autocorrect import Speller

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load stopwords and initialize spell checker
stop_words = set(stopwords.words("english"))
spell = Speller(lang='en')

def extract_text_from_pdf(pdf_path, start_page, end_page):
    PMI = fitz.open(pdf_path)
    text = ""
    for page_num in range(start_page, end_page):
        page = PMI.load_page(page_num)
        text += page.get_text("text")
    return text.replace("\n", "")

def remove_stop_words(sentence):
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def clean_text(text):
    # Cleaning steps as in your original code
    text = re.sub('[0-9]+.[0-9]+.[0-9]+.[0-9]+\t',' ',str(text))
    text = re.sub('[0-9]+.[0-9]+.[0-9]+\t',' ',str(text))
    text = re.sub('[0-9]+.[0-9]+\t',' ',str(text))
    text = re.sub('[[0-9]+.]+\t',' ',str(text))
    text = re.sub('[0-9]+.[0-9]+.[0-9]+.[0-9]+ ',' ',str(text))
    text = re.sub('[0-9]+.[0-9]+.[0-9]+ ',' ',str(text))
    text = re.sub('[0-9]+.[0-9]+ ',' ',str(text))
    text = re.sub('[[0-9]+.]+ ',' ',str(text))
    text = re.sub('[.[0-9]+]+\t',' ',str(text))
    text = re.sub('[.[0-9]+.]+ ',' ',str(text))
    text = re.sub('[0-9]+©2013 ',' ',str(text))
    text = re.sub('Project Management Institute.',' ',str(text))
    text = re.sub('Licensed To: Jorge Diego Fuentes Sanchez PMI MemberID: 2399412This copy is a PMI Member benefit, not for distribution, sale, or reproduction.','',str(text))
    text = re.sub('\n ',' ',str(text))
    text = re.sub('\n',' ',str(text))
    text = re.sub(':',' conceptualise ',str(text))
    text = re.sub('†','',str(text))
    text = re.sub("™","'",text)
    text = re.sub("¥","'",text)
    text = re.sub("'s","'",str(text))
    text = re.sub("-",' ',str(text))
    text = re.sub("— ",'',str(text))
    text = re.sub('\"',' ',str(text))
    text = re.sub('“',' ',str(text))
    text = re.sub("[\(\[].*?[\)\]]", " ", str(text))
    text = re.sub("[ ]+"," ",str(text))
    text = re.sub("˜","fi",text)
    text = re.sub("˚","kn",text)
    text = re.sub("Š"," ",text)
    text = re.sub(","," ",text)
    text = re.sub("PMBOK®","PMBOK",text)
    text = re.sub("INTRODUCTIONINTRODUCTIONA","A",text)
    text = re.sub("PROJECT RISK MANAGEMENTPROJECT RISK MANAGEMENT"," ",text)
    text = re.sub('A Guide to the Project Management Body of Knowledge Œ Fifth Edition 11 A Guide to the Project Management Body of Knowledge Œ Fifth Edition ','A Guide to the Project Management Body of Knowledge Œ Fifth Edition ',str(text))
    text = re.sub('A Guide to the Project Management Body of Knowledge Œ Fifth Edition ',' ',str(text))
    text = re.sub("PROJECT RI SK MANAGEMENTProject Risk Management"," ",text)
    text = re.sub("IDENTI FY RISKSIdentify Risks"," ",text)
    text = re.sub("PERFORM QUALI TATIVE RISK A NALYSISPerform Qualitative Risk Analysis"," ",text)
    text = re.sub("Part Guide PERFORM QUANTI TATIVE RISK A NALYSISPerform Quantitative Risk Analysis"," ",text)
    text = re.sub("PLAN RISK R ESPONSESPlan Risk Responses"," ",text)
    text = re.sub("Not For Distribution Sale or Reproduction"," ",text)
    text = re.sub("Control RisksControl Risks"," ",text)
    text = re.sub("IMPLEMENT RISK R ESPONSESImplement Risk Responses"," ",text)
    text = re.sub("MONI TOR RISKSMonitor Risks"," ",text)
    text = re.sub("Not For Distribution Sale or Reproduction.? ?"," ",text)
    text = re.sub("Not For Distribution Sale or Reproduction. "," ",text)
    text = re.sub("©2009 Project Management Institute. Practice Standard for Project Risk Management"," ",text)
    text = re.sub("project management institute"," ",text)
    text = re.sub("Practice Standard for Project Risk Management"," ",text)
    text = re.sub(r'[^.]*\binstitute\b[^.]*\.', '', text, flags=re.IGNORECASE)
    text = re.sub("  project management institute"," ",text)
    return text

def remove_special_characters(text):
    return text.translate(str.maketrans('', '', '™ŒŠ˚˜©®œ'))

def correct_spelling(sentence):
    return spell(sentence)

def process_text_data(pdf_path, start_page, end_page, output_path):
    # Extract text from PDF and create DataFrame
    text = extract_text_from_pdf(pdf_path, start_page, end_page)
    sentences = nltk.sent_tokenize(text)
    df = pd.DataFrame(sentences, columns=["sentence"])
    
    # Preprocess sentences
    df["sentence"] = df["sentence"].apply(remove_stop_words)
    df["sentence"] = df["sentence"].apply(clean_text)
    df["sentence"] = df["sentence"].apply(remove_special_characters)
    df["sentence"] = df["sentence"].apply(correct_spelling)
    df["sentence"] = df["sentence"].apply(lambda s: " ".join([word.text for word in nlp(s) if word.pos_ != "ADV"]))
    df["sentence"] = df["sentence"].str.lower()
    df = df[df['sentence'].str.strip() != ''].reset_index(drop=True)

    # Remove specific sentences and count
    sentence_to_remove = "project management institute"
    df = df[df['sentence'] != sentence_to_remove]
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

# Only run the process if the script is called directly (not imported)
if __name__ == "__main__":
    pdf_path = 'Projet_AI_Cognition/data/PMI_RM-standard.pdf'
    start_page = 37
    end_page = 67
    output_path = 'data/df_pmi_sent'  # Specify the output file path here
    process_text_data(pdf_path, start_page, end_page, output_path)
