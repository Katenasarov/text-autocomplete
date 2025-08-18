import pandas as pd
import re
import shutil
import os
import zipfile
import requests
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def download_and_extract_dataset():
    """Скачивает и извлекает датасет sentiment140"""
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    response = requests.get(url)
    with open("trainingdata.zip", "wb") as f:
        f.write(response.content)
    print("Архив скачан: trainingdata.zip")

    with zipfile.ZipFile("trainingdata.zip", 'r') as zip_ref:
        zip_ref.extractall("temp_data")
    print("Распаковано в папку temp_data")

    source = "temp_data/training.1600000.processed.noemoticon.csv"
    dest = "data/raw_dataset.csv"
    shutil.move(source, dest)
    print(f"Файл перемещён: {dest}")

    os.remove("trainingdata.zip")
    shutil.rmtree("temp_data")
    print("Временные файлы удалены")

def clean_text(text):
    """Очищает текст"""
    text = str(text).lower()
    text = re.sub(r'http[s]?://[^\s]+', '', text)   # ссылки
    text = re.sub(r'@[^\s]+', '', text)             # упоминания
    text = re.sub(r'[^a-z0-9\s]', ' ', text)        # спецсимволы
    text = re.sub(r'\s+', ' ', text).strip()        # лишние пробелы
    return text

def prepare_dataset():
    """Загружает, чистит и сохраняет датасет"""
    df = pd.read_csv("data/raw_dataset.csv", encoding='latin1',
                     names=['target', 'id', 'date', 'flag', 'user', 'text'])
    df['text'] = df['text'].astype(str).apply(clean_text)
    df = df[df['text'].str.len() > 1]
    df[['text']].to_csv("data/dataset_processed.csv", index=False)
    print("Очищенный датасет сохранён: data/dataset_processed.csv")

def split_dataset():
    """Разделение на train/val/test"""
    df = pd.read_csv("data/dataset_processed.csv")
    train, temp = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_STATE)

    train[['text']].to_csv("data/train.csv", index=False)
    val[['text']].to_csv("data/val.csv", index=False)
    test[['text']].to_csv("data/test.csv", index=False)

    print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")