from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import io

app = FastAPI()

model = None

class Data(BaseModel):
    feature1: float
    feature2: float
    # Add more features as needed
    # Ensure feature names match those used in your dataset

class ManualData(BaseModel):
    data: List[Dict[str, float]]

def train_model(df):
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    global model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    mse = train_model(df)
    return {"mean_squared_error": mse}

@app.post("/manual/")
async def manual_data(data: ManualData):
    df = pd.DataFrame(data.data)
    mse = train_model(df)
    return {"mean_squared_error": mse}

@app.post("/save-model/")
async def save_model():
    joblib.dump(model, 'house_price_model.pkl')
    return {"message": "Model saved successfully!"}

@app.post("/load-model/")
async def load_model():
    global model
    model = joblib.load('house_price_model.pkl')
    return {"message": "Model loaded successfully!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
