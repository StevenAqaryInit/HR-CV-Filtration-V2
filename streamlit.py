import streamlit as st
import pandas as pd
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import io
from docx import Document
import nltk
import re
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import AgGrid
# import openai
import concurrent.futures
import joblib




nltk.download('stopwords')
nltk.download('punkt')


titles = []
count_matrix = []
cos_similarity = []
indexes = []

stop_words = set(stopwords.words('english'))
vectorizer = CountVectorizer(stop_words='english')


# client = openai.OpenAI(api_key="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")


@st.cache
def get_main_phones(raw_phone_numbers):
    phone_pattern = re.compile(r'(?:(?:\d{3}-\d{7,8}|\d{2}-\d{7,8}|\+\d{2,3}-?\d{9,10}|\d{3}-\d{3}-\d{4}|\d{4,})|\+\d{12}|\d{2}-\d{5,6}-\d{4}|\d{14}|\d{2}-\d{9,10})')
    phone_numbers = [match.group() for match in phone_pattern.finditer(raw_phone_numbers)]
    for phone in phone_numbers:
        if len(phone) > 8:
            return phone


# @st.cache
def show(df):
    # data = df.iloc[:, 1:]
    try:
        gd = GridOptionsBuilder.from_dataframe(df.iloc[:, 1:])
        gd.configure_selection(selection_mode='single'
                            , use_checkbox=True)
        grid_options = gd.build()

        grid_table = AgGrid(df
                            , gridOptions=grid_options
                            , reload_data=False
                            , allow_unsafe_jscode=True
                            , height=500)
        
        values = list(grid_table['selected_rows'][0].values())[1:]
        keys = list(grid_table['selected_rows'][0].keys())[1:]
    
        record = {}
        for key, value in zip(keys, values):
            record[key] = value

        return record
    except:
        pass

    
@st.cache
def filter_text(text_content):
    tokens = word_tokenize(text_content)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    text = ' '.join(filtered_tokens)
    translation_table = str.maketrans('', '', string.punctuation)
    text = text.translate(translation_table)

    return text.lower()


@st.cache
def read_docx(file):
    doc = Document(file)
    text_content = ""
    for paragraph in doc.paragraphs:
        text_content += ' '.join(paragraph.text.split('\n'))

    cleaned_text = filter_text(text_content)
    return cleaned_text


@st.cache
def read_pdf(file):
    pdfReader = PyPDF2.PdfReader(file)

    pages_list = ''
    for page in pdfReader.pages:
        pages_list += ' '.join(page.extract_text().split('\n'))

    cleaned_text = filter_text(pages_list)
    return cleaned_text


@st.cache
def get_files_content(cv):
    file_bytes = b''
    # nationality = postition = phone_number = ''

    if cv.name.split('.')[-1] == 'pdf':
        file_bytes = io.BytesIO(cv.read())
        cleaned_text = read_pdf(file_bytes)

    elif cv.name.split('.')[-1] == 'docx' or cv.split('.')[-1] == 'doc':
        file_bytes = io.BytesIO(cv.read())
        cleaned_text = read_docx(file_bytes)


    model = joblib.load("./classifiers/models/job_title_model.sav")
    vectorizer = joblib.load("./classifiers/vectorizers/job_title_vectorizer.pk")

    x_data = vectorizer.transform([cleaned_text])
    postition = model.predict(x_data)[0]

    # postition = chat(cleaned_text.lower(), '''
    # what is the postition of this person ?.  give me the position only without any comments from you
    # ''')

    # nationality = chat(cleaned_text.lower(), '''
    # what is the nationality of this person ?.  give me the nationality only without any comments from you
    #     ''')

    phone_number = get_main_phones(cleaned_text.lower())

    return postition, phone_number, file_bytes, cleaned_text


# @st.cache
def go_to_threads(files):
    df = pd.DataFrame()
    files_bytes = []
    position_list = []
    phone_number_list = []
    # nationality_list = []


    for cv in files:
        print(cv)
        print(cv.name)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            thread = executor.submit(get_files_content, (cv))
            result_value = thread.result()
            
            position_list.append(result_value[0])
            phone_number_list.append(result_value[1])
            files_bytes.append(result_value[2])
            cleaned_text =  result_value[3]
            titles.append(cv.name)
            df = pd.concat([df, pd.DataFrame([cleaned_text.lower()], columns=['cv'])], ignore_index=True)
            
            # print('--------------------------')

    df['title'] = titles
    df['bytes'] = files_bytes
    df['postition'] = position_list
    df['phone_number'] = phone_number_list
    # df['nationality'] = nationality_list

    return df


# @st.cache
def get_scores(cv, clean_keywords):
    matrix = vectorizer.fit_transform([cv, clean_keywords])
    scores = cosine_similarity(matrix)[0][1]

    return scores



st.title('CV Filteration')

files = st.file_uploader('Upload files', accept_multiple_files=True, type=['PDF', 'DOCX', 'DOC'])
keywords = st.text_area('Enter the keywords here...')
contents = []
scores = []

df = pd.DataFrame()

df = go_to_threads(files)
clean_keywords = filter_text(keywords)


if clean_keywords != '':
    for cv in df.iloc[:, 0].values.tolist():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            thread = executor.submit(get_scores, cv, clean_keywords)
            result_value = thread.result()
            scores.append(result_value)
        
    df['scores'] = scores
    df['scores'] = df['scores'] * 100

    record = show(df[['bytes', 'title', 'scores', 'postition', 'phone_number']].sort_values(by='scores', ascending=False).drop_duplicates())
    try:
        row = df[df['title']==record['title']]
        print(row)
    except:
            pass


    try:
        for file in files:
            if file.name == record['title']:
                st.download_button(
                    label="Download file",
                    data=bytes(file.getbuffer()),
                    file_name=row['title'].values[0],
                )
    except:
        pass

else:
    st.error('Enter Some Keywords...')






