from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

import pandas as pd

from NeuralNetwork import predict

classes = pd.read_excel("./meta/Classes_Description.xls")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return HTMLResponse(content=open('./static/main.html').read())


@app.post("/image")
async def upload_file(img: UploadFile):
    print("Got request")
    try:
        content = await img.read()
    except Exception as e:
        print(e)
        return {"message": "Invalid Base64 String"}
    prediction = predict(content)
    num_class = pd.Series(prediction[0]).idxmax()
    sign_name = classes["Descriptive name"].iloc[num_class]

    return {"message": f"{sign_name}"}
