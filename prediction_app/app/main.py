from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .model import load_models, predict_title

app = FastAPI()

# 設定靜態檔案和模板
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# 載入模型
doc2vec_model, classifier_model = load_models()


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
