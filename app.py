from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

#templates
templates = Jinja2Templates(directory="templates")

with open("car_price_model.pkl", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    encoders = saved_data["encoders"]
    scaler = saved_data["scaler"]

# === Swagger input model ===
class CarPrice(BaseModel):
    Make: str
    Year: int
    EngineFuelType: str
    EngineHP: float
    EngineCylinders: float
    TransmissionType: str
    Driven_Wheels: str
    Doors: float
    VehicleSize: str
    VehicleStyle: str
    highwayMPG: int
    cityMPG: int

# ✨ Eğitim verisindeki kolon isimlerine dönüştürmek için mapping
column_mapping = {
    "Make": "Make",
    "Year": "Year",
    "EngineFuelType": "Engine Fuel Type",
    "EngineHP": "Engine HP",
    "EngineCylinders": "Engine Cylinders",
    "TransmissionType": "Transmission Type",
    "Driven_Wheels": "Driven_Wheels",
    "Doors": "Number of Doors",
    "VehicleSize": "Vehicle Size",
    "VehicleStyle": "Vehicle Style",
    "highwayMPG": "highway MPG",
    "cityMPG": "city mpg"
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/options")
async def get_options():
    """Kategorik değerleri ve sayısal alanların min/max değerlerini döndürür"""
    df = pd.read_csv("car_data.csv")
    
    return {
        "Make": sorted([x for x in df["Make"].dropna().unique() if pd.notna(x)]),
        "EngineFuelType": sorted([x for x in df["Engine Fuel Type"].dropna().unique() if pd.notna(x)]),
        "TransmissionType": sorted([x for x in df["Transmission Type"].dropna().unique() if pd.notna(x)]),
        "Driven_Wheels": sorted([x for x in df["Driven_Wheels"].dropna().unique() if pd.notna(x)]),
        "VehicleSize": sorted([x for x in df["Vehicle Size"].dropna().unique() if pd.notna(x)]),
        "VehicleStyle": sorted([x for x in df["Vehicle Style"].dropna().unique() if pd.notna(x)]),
        "Doors": sorted([float(x) for x in df["Number of Doors"].dropna().unique() if pd.notna(x)]),
        "Year": {"min": int(df["Year"].min()), "max": int(df["Year"].max())},
        "EngineHP": {"min": float(df["Engine HP"].min()), "max": float(df["Engine HP"].max())},
        "EngineCylinders": {"min": float(df["Engine Cylinders"].min()), "max": float(df["Engine Cylinders"].max())},
        "highwayMPG": {"min": int(df["highway MPG"].min()), "max": int(df["highway MPG"].max())},
        "cityMPG": {"min": int(df["city mpg"].min()), "max": int(df["city mpg"].max())}
    }

@app.post("/predict")
async def predict(features: CarPrice):
    # Dict → DataFrame
    df = pd.DataFrame([features.model_dump()])

    # Kolon isimlerini modele uygun hale getir
    df = df.rename(columns=column_mapping)

    # Encoder uygulanmış kategorik dönüşüm
    df_encoded = encoders.transform(df)

    # Skaler uygula
    df_scaled = scaler.transform(df_encoded)

    # Tahmin üret
    prediction = model.predict(df_scaled)

    return {"predicted_price": float(prediction[0])}
