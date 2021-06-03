from flask import Flask, request, jsonify
from flask_cors import CORS
from processing import *
import torch
from kobert_transformers import get_tokenizer, get_kobert_model
from news_model.news_dataset import NewsDataset
from news_model.pred import Pred
from torch.utils.data import DataLoader

data_example = {"news": [
    {
        "company": "YG",
        "title": "손나은 YG엔터 에이핑크 어떻게 될까",
        "link": "https://www.donga.com/news/Entertainment/article/all/20210504/106753784/1",
        "text": "6인 걸그룹 에이핑크 멤버 손나은이 YG엔터테인먼트로 옮기면서 팀 활동이 이어질지 관심을끈다.최근 에이핑크의 다른 멤버들인 박초롱 윤보미 정은지 김남주 오하영 등은 자신들을 발굴한플레이엠엔터테인먼트와 재계약을 체결했다.다만 손나은은 배우 활동에 힘을 싣겠다며 김희애 차승원·강동원이 속한 YG엔터테인먼트에 새로 둥지를 틀었다.에이핑크는 지난달 데뷔 10주년을 맞았다. 미스터 츄 노노노 러브 등의 히트곡을 남겼다. 주로 청순한면모를선보이다가 1도 없어 덤더럼으로 콘셉트 변주에 성공했다. 멤버 각자가 솔로 앨범 연기 예능 광고 등 다양한분야에서 활약 중이다",
        "createdAt": "2021-05-04 06:23"
    },
    {
        "company": "YG",
        "title": "문 대통령, 시민 모욕죄 고소 취소…감내해야 한다는 지적 수용",
        "link": "https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=100&oid=028&aid=0002543167",
        "text": "문재인 대통령이 자신과 가족들을 비난한 30대 남성을 모욕죄 등으로 2년 전 고소한 사건을 취소하기로 \\n했다. 친고죄인 모욕죄는 피해자가 처벌 의사를 철회하면 기소할 수 없다. 최고 권력자인 대통령이 시민 개인을 처벌해\\n달라청원하는 것은 지나치다는 시민사회와 야당의 지적을 받아들인 것으로 보인다.\n박경미 청와대 대변인은 4일 문재인 대통령은 2019년 전단 배포에 의한 모욕죄와 관련하여 처벌 의사를 철회하도록지시했\\n다고 밝혔다. 주권자인 국민의 위임을 받아 국가를 운영하는 대통령으로서 모욕적인 표현을 감내하는 것도 필요하다는지\\n적을 수용하여, 이번 사안에 대한 처벌 의사 철회를 지시한 것이라고 박경미 대변인은 전했다. 문 대통령은 이어 앞으로\\n명백한 허위사실을 유포하여 정부에 대한 신뢰를 의도적으로 훼손하고, 외교적 문제로 비화될 수 있는 행위에 대해서는 적어도 사실관계를 바로잡는다는 취지에서 개별 사안에 따라 신중하게 판단하여 결정할 예정이라고 했다.\\n앞서 문 대통령은 지난 2019년 국회 분수대 앞에서 자신과 박원순 전 서울시장, 유시민 노무현재단 이사장, 홍영표 전더불\\n어민주당 원내대표의 선친이 친일을 했다는 내용 등을 담은 전단지를 배포한 김아무개씨를 대리인을 통해 고소했다.전단지\\n 뒷면에는 문 대통령을 모욕하는 여러 문구들이 적혔다. 박 대변인은 이날 대통령은 본인과 가족들에 대해 차마 입에담기 \\n어려운 혐오스러운 표현도, 국민들의 표현의 자유를 존중하는 차원에서 용인해 왔다. 그렇지만 이 사안은 대통령 개인에대\\n한 혐오와 조롱을 떠나, 일본 극우주간지 표현을 무차별적으로 인용하는 등 국격과 국민의 명예, 남북관계 등 국가의미래\\n에 미치는 해악을 고려해 대응했던 것이라며 고소를 결정했던 배경을 설명했다.\\n대통령이 처벌의사를 철회한 모욕죄 고소 논란은 경찰이 수사를 끝내고 최근 기소 의견으로 검찰로 사건을 보내면서 다시\\n수면 위로 올라왔다. 당초 청와대는 김씨 고소 사실을 공개하지 않았으며, 전단을 뿌린 김씨가 반성 의사도 없다는 점을\\n들어 처벌 의사를 철회하기 어렵다는 입장이었다.\\n그러나 참여연대는 지난 3일 논평을 내어 권력에 대한 국민의 비판을 모욕죄로 처벌하는 것은 문 대통령이 그간 밝힌국정\\n철학과도 맞지 않다며 고소 취소를 촉구했다. 지난해 8월 정부를 비난하거나 대통령을 모욕하는 정도는 표현의 범주로허\\n용해도 된다. 대통령 욕해서 기분이 풀리면 그것도 좋은 일이라고 했던 문 대통령의 발언과도 배치된다는 지적이었다.청\\n년정의당도 독재국가에서는 대통령에 대한 모욕이 범죄일지 모르지만, 민주주의 국가에서 대통령이라는 위치는 모욕죄가\\n성립되어선 안되는 대상이라며 배포된 내용이 어떤 것이었든, 대통령에 의한 시민 고소는 부적절하다고 주장했다.\\n문 대통령은 처벌의사 철회를 계기로 국격과 국민의 명예, 국가의 미래에 악영향을 미치는 허위사실 유포에 대한 성찰의\\n계기가 되기를 바란다고 밝혔다. 청와대 관계자는 문 대통령이 2년 전 고소를 이제 철회하는 이유에 대해 “2019년전단은\\n 대한민국 대통령을 북한의 개라고 조롱한, 그런 도를 넘어서는 전단이었다. 정말 혐오스러운 표현들이 있기는 하지국가\\n를 운영하는 대통령으로서 감내하겠다는 뜻”이라고 설명했다.",
        "createdAt": "2021-05-04-17:06"
    }]}

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
        success_check_type = False

        if type(news_dto) == list:
            success_check_type = True
            for news in news_dto:
                if not check_type(news_dto_type, news):
                    success_check_type = False
                    break

        if not success_check_type:
            return '', 400

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
        ]
        return jsonify(result)
    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    LOAD_PATH = ''
    checkpoint = torch.load(LOAD_PATH)
    tokenizer = get_tokenizer()
    model = Pred(checkpoint)
    get_dataloader = lambda texts: DataLoader(NewsDataset(texts, [None] * len(texts), tokenizer,
                                                          checkpoint['hyper_params']['embed_size'],
                                                          checkpoint['hyper_params']['batch_size']),
                                              batch_size=checkpoint['hyper_params']['batch_size'])
    app.run(debug=True)
