from typing import Union

from fastapi import FastAPI, File
from neural_style import convert_cnn
from gan import convert_gan
import requests

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/image/convert/{convert_tag}")
async def read_user_item(
    convert_tag: str,
    origin_img: bytes = File(),
    style_img: Union[bytes, None] = None,
    token: Union[str, None] = None,
):
    if convert_tag == "cnn":
        result_img = await convert_cnn(origin_img, style_img)
    else:
        result_img = await convert_gan(convert_tag, origin_img, token)

    file = {'result_img': open(result_img, 'rb')}
    response = requests.post("https://missulgan.art/images/upload", file)

    return response.filename