from typing import Union

from fastapi import FastAPI, File, UploadFile, Form
from neural_style import convert_cnn
from gan import convert_gan
import requests

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/image/convert")
async def read_user_item(
    origin_img: bytes = File(),
    convert_tag: str = Form(...),
    style_img: Union[bytes, None] = None,
    token: Union[str, None] = Form(...),
):
    #origin_img = await origin_img.read()
    if convert_tag == "cnn":
        result_img = await convert_cnn(origin_img, style_img)
    else:
        if convert_tag == "8":
            convert_tag = "style_vangogh_pretrained"
        elif convert_tag == "9":
            convert_tag = "style_monet_pretrained"
        elif convert_tag == "10":
            convert_tag = "style_cezanne_pretrained"
        elif convert_tag == "11":
            convert_tag = "style_ukiyoe_pretrained"

        result_img = await convert_gan("style_vangogh_pretrained", origin_img, '/'+token)

    file = {'file': open(result_img, 'rb')}
    access_token = 'eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJwYXJrY2hlb2xAa2FrYW8uY29tIiwicm9sZSI6IlJPTEVfVVNFUiIsImlhdCI6MTY1OTUwODMwNCwiZXhwIjoxNjY5NTk0NzA0fQ.jaRKOC2eOS0y4skRiWdu65R4qAZnsse9Y441_zJPLAqawQlxxgHvc4VsSWKu_G4UyJGPMvDz1FB_zb1rua9jlQ'
    auth_header = {'Authorization': 'Bearer ' + access_token}
    response = requests.post("https://api.missulgan.art/image/upload", file, headers = auth_header)
    return response.filename
