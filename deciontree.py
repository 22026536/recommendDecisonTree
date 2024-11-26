from fastapi import FastAPI, Request
import pandas as pd
from pymongo import MongoClient

# Khởi tạo app
app = FastAPI()

client = MongoClient("mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/web_project")
db = client["anime_tango2"]

# Tải dữ liệu từ MongoDB
df_ratings = pd.DataFrame(list(db["UserRating"].find()))
df_anime = pd.DataFrame(list(db["Anime"].find()))


############################
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Chuẩn bị dữ liệu cho Decision Tree và Naive Bayes
label_encoder = LabelEncoder()
df_anime['Name_encoded'] = label_encoder.fit_transform(df_anime['Name'])  # Mã hóa tên anime

# Kết hợp thông tin xếp hạng vào dữ liệu phim
df_merged = df_ratings.merge(df_anime, on="Anime_id")
X = df_merged[['User_id', 'Rating']]  # Đặc trưng đầu vào
y = df_merged['Name_encoded']         # Nhãn đầu ra

# Tách tập dữ liệu thành train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mô hình Decision Tree
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)


def recommend_by_decision_tree(user_id, n):
    """
    Gợi ý phim sử dụng mô hình Decision Tree.
    """
    user_ratings = df_ratings[df_ratings['User_id'] == user_id]
    if user_ratings.empty:
        return {"error": "Người dùng không có đánh giá."}
    
    predictions = decision_tree_model.predict(user_ratings[['User_id', 'Rating']])
    recommended_names = label_encoder.inverse_transform(predictions)
    return [{"Name": name} for name in recommended_names[:n]]


# Endpoint cho /recommend2
@app.post("/")
async def deciontree(request: Request):
    data = await request.json()
    user_id = data.get("User_id")
    n = data.get("n", 10)  # Số lượng gợi ý mặc định là 10

    if user_id is None:
        return {"error": "Vui lòng cung cấp user_id"}

    # Gọi hàm recommend_by_decision_tree
    result = recommend_by_decision_tree(user_id, n)
    return {"recommendations": result}

import uvicorn
import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render sẽ cung cấp cổng trong biến PORT
    uvicorn.run("deciontree:app", host="0.0.0.0", port=port)

