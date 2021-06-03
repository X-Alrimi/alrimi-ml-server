from flask import Flask, request, jsonify
from flask_cors import CORS
from processing import *

app = Flask(__name__)
CORS(app)


def check_type(data_type, data):
    for key in data_type:
        if not (key in data and type(data[key]) == data_type[key]):
            return False
    return True


@app.route('/api/predict', methods=['POST'])
def predict_route():
    try:
        news_dto_type = {'company': str, 'title': str, 'link': str, 'text': str, 'createdAt': str}
        news_dto = request.json
        if not check_type(news_dto_type, news_dto):
            return '', 400

        text = preprocessing(news_dto['text'])
        _ = inference(model, text)

        result = {
            'news': {key: news_dto[key] for key in news_dto if key != 'text'},
            'isCritical': False,
            'similarity': []
        }
        return jsonify(result)
    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    # model = load_model('')
    model = DummyModel()
    app.run(debug=True)
