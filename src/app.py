import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(data):
    # Giả sử cột 'satisfaction' là mục tiêu 
    # Thực hiện mã hóa và xử lý dữ liệu nếu cần
    return data.drop('satisfaction', axis=1), data['satisfaction']


def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def visualize_data(data):
    plt.figure(figsize=(10,6))
    data['satisfaction'].value_counts().plot(kind='bar')
    plt.title('Phân phối sự hài lòng của khách hàng')
    plt.xlabel('Mức độ hài lòng')
    plt.ylabel('Số lượng')
    plt.show()


def main():
    data = load_data('customer_feedback.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    visualize_data(data)


if __name__ == '__main__':
    main()
