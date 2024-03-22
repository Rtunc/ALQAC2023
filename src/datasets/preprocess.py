import re
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_text(text):
    # Loại bỏ ký tự đặc biệt và chuyển đổi thành chữ thường
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    
    # Sử dụng thư viện pyvi để tách từ với tiếng Việt
    tokens = ViTokenizer.tokenize(text)
    return tokens
def prepairdata(df):
    sen1 = []
    sen2 = []
    labels = []
    for index, row in df.iterrows():
        sen1.append(row['text1'])
        sen2.append(preprocess_text(row['text2']))
        labels.append(row['label'])  
   
    return sen1, sen2, labels