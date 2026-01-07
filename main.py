#test deploy 
from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware

import firebase_admin
from firebase_admin import credentials, firestore

app = FastAPI()   # ✅ THIS WAS MISSING

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev ke liye OK
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Firebase init
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

@app.get("/attendance-logs")
def get_attendance_logs():
    docs = (
        db.collection("attendance")
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(20)
        .stream()
    )

    logs = []
    for d in docs:
        data = d.to_dict()
        logs.append({
            "name": data.get("name"),
            "status": data.get("status"),
            "time": data.get("time"),
            "date": data.get("date"),
            "confidence": data.get("confidence"),
        })

    return {"logs": logs}
