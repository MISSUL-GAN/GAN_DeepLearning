from flask import Flask
from flask import request
from neural_style import convert_cnn
from gan import convert_gan
import requests
import io
from PIL import Image

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

    result_img.save('./result.jpg', 'JPEG') 
    file = {'file': open('result.jpg', 'rb')}
    access_token = 'eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJwYXJrY2hlb2xAa2FrYW8uY29tIiwicm9sZSI6IlJPTEVfVVNFUiIsImlhdCI6MTY1OTUwODMwNCwiZXhwIjoxNjY5NTk0NzA0fQ.jaRKOC2eOS0y4skRiWdu65R4qAZnsse9Y441_zJPLAqawQlxxgHvc4VsSWKu_G4UyJGPMvDz1FB_zb1rua9jlQ'
    auth_header = {'Authorization': 'Bearer ' + access_token}
    response = requests.post("https://api.missulgan.art/image/upload", file, headers = auth_header)
    print(response)
    return response.fileName

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
