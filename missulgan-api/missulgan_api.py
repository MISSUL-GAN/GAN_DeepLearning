from flask import Flask
from flask import request
from neural_style import convert_cnn
from gan import convert_gan
import requests
import io
from PIL import Image
import json
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return {"message": "Hello World"}

@app.route("/image/convert", methods=["POST"])
def convert_img():
    origin_img = request.files['origin_img']
    style_img = request.files['style_img']
    convert_tag = request.form['convert_tag']
    token = request.form['token']
    #origin_img = await origin_img.read()
    if convert_tag == "cnn":
        result_img = convert_cnn(origin_img, style_img)
        result_img = Image.fromarray(np.uint8(result_img))
    else:
        if convert_tag == "8":
            convert_tag = "style_vangogh_pretrained"
        elif convert_tag == "9":
            convert_tag = "style_monet_pretrained"
        elif convert_tag == "10":
            convert_tag = "style_cezanne_pretrained"
        elif convert_tag == "11":
            convert_tag = "style_ukiyoe_pretrained"

        result_img = convert_gan(convert_tag, origin_img, './'+token)

    result_img.save('./result.jpg','JPEG')
    access_token = 'eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJwYXJrY2hlb2xAa2FrYW8uY29tIiwicm9sZSI6IlJPTEVfVVNFUiIsImlhdCI6MTY1OTUwODMwNCwiZXhwIjoxNjY5NTk0NzA0fQ.jaRKOC2eOS0y4skRiWdu65R4qAZnsse9Y441_zJPLAqawQlxxgHvc4VsSWKu_G4UyJGPMvDz1FB_zb1rua9jlQ'
    auth_header = {'Authorization': 'Bearer ' + access_token}
    multipart_form_data = {'file': ('result.jpg', open('result.jpg', 'rb'), 'image/jpeg')}
    response = requests.post('https://api.missulgan.art/image/upload', files=multipart_form_data, headers = auth_header)
    
    response_json = json.loads(response.text)
    return response_json
    #print(response_json)
    #return response_json['fileName']

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
