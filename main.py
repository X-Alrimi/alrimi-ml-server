from flask import Flask, request, jsonify
from flask_cors import CORS
from processing import *
import torch
from kobert_transformers import get_tokenizer, get_kobert_model
from news_model.news_dataset import NewsDataset
from news_model.pred import Pred
from torch.utils.data import DataLoader

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

        test_dataset = NewsDataset(test_data['input'], test_data['label'], tokenizer, \
                                   checkpoint['hyper_params']['embed_size'], \
                                   checkpoint['hyper_params']['batch_size'])

        test_dataloader = DataLoader(test_dataset, batch_size=checkpoint['hyper_params']['batch_size'], \
                                     num_workers=checkpoint['hyper_params']['batch_size'])

        text = preprocessing(news_dto['text'])
        preds, vecs = model(test_dataloader)

        result = {
            'news': {key: news_dto[key] for key in news_dto if key != 'text'},
            'isCritical': False,
            'similarity': []
        }
        return jsonify(result)
    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    LOAD_PATH = ''
    checkpoint = torch.load(LOAD_PATH)
    tokenizer = get_tokenizer()
    model = Pred(checkpoint)

    app.run(debug=True)
