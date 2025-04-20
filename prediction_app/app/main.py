import csv
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .model import load_models, predict_title


class FeedbackRequest(BaseModel):
    title: str
    label: str


app = FastAPI()

# 設定靜態檔案和模板
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# 載入模型
doc2vec_model, classifier_model = load_models()

# 確保 CSV 檔案存在
FEEDBACK_CSV = Path(__file__).parent / "user-labeled-titles.csv"
if not FEEDBACK_CSV.exists():
    with open(FEEDBACK_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "label"])


# 首頁路由
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 預測 API
@app.get("/api/model/prediction")
async def predict_board(title: str):
    if not title:
        raise HTTPException(status_code=400, detail="標題不能為空")

    # 使用模型進行預測
    result = predict_title(title, doc2vec_model, classifier_model)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return JSONResponse(content=result)


# 反饋 API
@app.post("/api/model/feedback")
async def save_feedback(feedback: FeedbackRequest):
    if not feedback.title or not feedback.label:
        raise HTTPException(status_code=400, detail="標題和標籤不能為空")

    try:
        with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([feedback.title, feedback.label])
        return JSONResponse(content={"status": "success", "message": "OK"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
