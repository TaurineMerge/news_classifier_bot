import os
import shutil
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from newsbot.asr import WhisperASR
from newsbot.tfidf_classifier import TfidfClassifier

app = FastAPI(title="NLP Audio Classifier API")

asr = WhisperASR()
classifier = TfidfClassifier()

LABEL_MAP = {
    "спорт": 0,
    "технологии": 1,
    "политика": 2,
    "экономика": 3,
    "происшествия": 4,
    "культура": 5,
    "другое": 6,
}


@app.post("/predict-audio")
async def predict_audio(file: Annotated[UploadFile, File()]):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        transcription = asr.transcribe(temp_path)
        text = transcription.text

        if not text or not text.strip():
            return {"label": -1, "text": "Не удалось распознать речь"}

        prediction = classifier.predict(text)
        label_int = LABEL_MAP.get(prediction.topic, -1)

        return {
            "label": label_int,
            "topic": prediction.topic,
            "text": text,
            "confidence": prediction.confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}
