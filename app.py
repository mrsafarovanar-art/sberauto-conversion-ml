"""Локальный сервис: uvicorn app:app --host 127.0.0.1 --port 8035."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from predict import Predictor
from schema import SessionInput, BatchInput


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = Predictor()
    yield


app = FastAPI(title='СберАвтоподписка · прогноз целевого действия', version='2.0.0', lifespan=lifespan)


@app.get('/', response_class=HTMLResponse)
def home():
    return '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Прогноз целевого действия</title>
    <style>body{font:18px system-ui;max-width:760px;margin:70px auto;padding:20px;color:#19302c}textarea{width:100%;height:250px;font:15px monospace;padding:12px;box-sizing:border-box}button{padding:12px 22px;background:#126957;color:white;border:0;border-radius:7px;font-size:18px;margin-top:14px}pre{white-space:pre-wrap;background:#edf5f2;padding:20px}a{color:#126957}</style>
    <h1>СберАвтоподписка</h1><p>Прогноз целевого действия по характеристикам визита</p>
    <p>Введите JSON с признаками utm_*, device_* и geo_*. Ответ: 1 — прогнозируется целевое действие, 0 — не прогнозируется.</p>
    <textarea id="input" aria-label="Признаки визита">{"utm_medium":"organic","device_category":"mobile","device_os":"Android","device_screen_resolution":"393x851","geo_country":"Russia","geo_city":"Moscow"}</textarea>
    <button id="run">Получить прогноз</button><pre id="output" aria-live="polite">Готов к работе</pre>
    <p><a href="/docs">Документация API</a> · Учебная модель на данных 2021 года. Прогноз не гарантирует действие пользователя.</p>
    <script>document.getElementById('run').onclick=async()=>{const out=document.getElementById('output');try{const data=JSON.parse(document.getElementById('input').value);out.textContent='Расчёт…';const r=await fetch('/predict_proba',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});out.textContent=JSON.stringify(await r.json(),null,2)}catch(e){out.textContent='Ошибка: '+e.message}};</script></html>'''


@app.get('/health')
def health(request: Request):
    return {'status': 'ok', 'model': request.app.state.predictor.metadata['selected_model']}


@app.post('/predict', response_model=int)
def predict(session: SessionInput, request: Request):
    """Число 0 или 1 в JSON, как требуется в задании."""
    return request.app.state.predictor.predict([session.model_dump()])[0]


@app.post('/predict_proba')
def predict_proba(session: SessionInput, request: Request):
    predictor = request.app.state.predictor
    p = float(predictor.probabilities([session.model_dump()])[0])
    return {'prediction': int(p >= predictor.threshold), 'probability': p, 'threshold': predictor.threshold}


@app.post('/predict_batch', response_model=list[int])
def predict_batch(batch: BatchInput, request: Request):
    return request.app.state.predictor.predict([row.model_dump() for row in batch.sessions])
