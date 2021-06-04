from flask import Flask, request, jsonify
from flask_cors import CORS
from processing import *
import torch
from kobert_transformers import get_tokenizer, get_kobert_model
from news_model.news_dataset import NewsDataset
from news_model.pred import Pred
from torch.utils.data import DataLoader
from waitress import serve
import traceback

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
        news_dto = request.json['news']
        success_check_type = False

        if type(news_dto) == list:
            success_check_type = True
            for news in news_dto:
                if not check_type(news_dto_type, news):
                    success_check_type = False
                    break

        if not success_check_type:
            return 'Type is wrong!', 400

        texts = []
        for news in news_dto:
            texts.append(preprocessing(news['text']))

        test_dataloader = get_dataloader(texts)
        preds, vecs = model(test_dataloader)

        result = [
            {
                'news': {key: news[key] for key in news if key != 'text'},
                'isCritical': critical,
                'similarity': vector
            }
            for news, critical, vector in zip(news_dto, preds, vecs)
            if critical == 1
        ]
        return jsonify(result)
    except Exception as e:
        print(traceback.format_exc())
        return 'Sorry, Something is wrong.', 500


if __name__ == '__main__':
    LOAD_PATH = './model.pth'
    print('Start load model')
    checkpoint = torch.load(LOAD_PATH)
    tokenizer = get_tokenizer()
    model = Pred(checkpoint)
    get_dataloader = lambda texts: DataLoader(NewsDataset(texts, [0] * len(texts), tokenizer,
                                                          checkpoint['hyper_params']['embed_size'],
                                                          1),
                                              batch_size=1)
    # app.run(debug=True)
    print('Starting server at 0.0.0.0:5000')
    serve(app, host='0.0.0.0', port=5000)
