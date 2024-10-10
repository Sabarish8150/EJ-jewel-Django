import requests
import csv
import pandas as pd
import numpy as np
import cv2
import pytesseract
from django.shortcuts import render
from .forms import UploadImageForm
from PIL import Image

# Load Google Sheet as CSV
sheet_url = "https://docs.google.com/spreadsheets/d/1OMN8Fdby7qCugZeNg1Ix6IL6_abSuIuszGvi7Qrj2w8/pub?gid=0&single=true&output=csv"
response = requests.get(sheet_url)
data = response.content.decode('utf-8')

# Parse CSV data
csv_data = list(csv.reader(data.splitlines()))
df = pd.DataFrame(csv_data)
df = df.drop(index=1)
df = df.drop(index=0)

# Extract non-null product names from column 13 (12th index)
products = df.iloc[:, 12].dropna().tolist()
products = [x for x in products if x.strip() != '']

def is_numeric(value):
    """Utility function to check if a value can be converted to a float or int."""
    # try:
    #     float(value)  # Try converting to a float
    return True
    # except ValueError:
        # return False

def total_pieces(total):
    global df
    total_sum = 0
    for idx, row in df.iterrows():
        from_val = row.iloc[4]  
        to_val = row.iloc[5]  

        if is_numeric(from_val) and is_numeric(to_val):
            from_val = int(from_val)
            to_val = int(to_val)
            grade_range = float(row.iloc[6])  

            if from_val <= total <= to_val: 
                total_sum += grade_range  
                break 

    return total_sum

def gold_weight(gold_wt):
    global df
    total_sum = 0
    for idx, row in df.iterrows():
        gold_weight_from = row.iloc[0]  
        gold_weight_to = row.iloc[1]   

        if is_numeric(gold_weight_from) and is_numeric(gold_weight_to):
            gold_weight_from = float(gold_weight_from)
            gold_weight_to = float(gold_weight_to)
            grade_range = float(row.iloc[2])  
            if gold_weight_from <= gold_wt <= gold_weight_to: 
                total_sum += grade_range  
                break  

    return total_sum

def Sur(S_area):
    global df
    total_sum = 0
    for idx, row in df.iterrows():
        gold_weight_from = row.iloc[8] 
        gold_weight_to = row.iloc[9]   

        if not is_numeric(gold_weight_from) or not is_numeric(gold_weight_to):
            continue
        else:
            gold_weight_from = float(gold_weight_from)
            gold_weight_to = float(gold_weight_to)
            grade_range = float(row.iloc[10])  
            
            if gold_weight_from <= S_area <= gold_weight_to:  
                total_sum += grade_range  
                break 

    return total_sum

def jewel_type(total, gold_wt, j_type, mode):
    for idx, row in df.iterrows():
        if row.iloc[12] == j_type:  
            if mode == "Array":
                return int(total) * int(row.iloc[13])/100 , float(gold_wt) * int(row.iloc[13])/100
            elif mode == "Mirror":
                return int(total) * int(row.iloc[14]) / 100 , float(gold_wt) * int(row.iloc[14]) / 100
    return 0, 0

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh

def extract_table_data(image):
    h, w, _ = image.shape
    cropped_image = image[int(h * 0.75):h, int(w * 0.3):w]
    preprocessed_image = preprocess_image(cropped_image)
    extracted_text = pytesseract.image_to_string(preprocessed_image)

    # Clean up the text
    extracted_text = extracted_text.replace("|", " ").strip()
    extracted_text = extracted_text.replace(";", " ").strip()
    extracted_text = extracted_text.replace("[", " ").strip()
    extracted_text = extracted_text.replace("]", " ").strip()
    total, gold_wt = None, None
    lines = extracted_text.split('\n')
    S_area = 0

    for line in lines:
        line_split = line.split()

        if 'Total' in line or 'TOTAL' in line:
            total = line.split()[-2] 
        if 'Gold Wt' in line or 'Gold' in line:
            gold_wt = line.split()[2] 
        if not is_numeric(gold_wt):
            gold_wt = 0
        if 'Surface' in line or 'surface' in line:
            S_area = line_split[2]

    return total, gold_wt, extracted_text, S_area

def index(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = request.FILES['image']
            image = np.array(Image.open(uploaded_image))
            total, gold_wt, extracted_text, S_area = extract_table_data(image)

            jewelry_type = request.POST.get('jewelry_type')
            mode = request.POST.get('mode')

            if total and gold_wt:
                total = int(total)
                gold_wt = float(gold_wt)

                if mode == "Mirror":
                    S_area = float(S_area) if S_area else 0
                
                total, gold_wt = jewel_type(total, gold_wt, jewelry_type, mode)

                total_sum = 0
                count = 1
                if mode == "Array":
                    total_sum += total_pieces(total)
                    total_sum += gold_weight(gold_wt)
                    count = 2
                if mode == "Mirror":
                    total_sum += total_pieces(total)
                    total_sum += gold_weight(gold_wt)
                    total_sum += Sur(S_area)
                    count = 3

                grade = round(total_sum / count, 1)

                return render(request, 'result.html', {
                    'total': total,
                    'gold_wt': gold_wt,
                    'S_area': S_area,
                    'grade': grade,
                    'jewelry_type': jewelry_type,
                    'mode': mode,
                })
    else:
        form = UploadImageForm()
    
    return render(request, 'index.html', {'form': form, 'products': products})

