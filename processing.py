import re


def clean_text(text):
    review = re.sub(r'[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣|\s]', ' ', str(text))  # remove punctuation
    review = review.strip()
    review = re.sub(r'\s+', ' ', review)  # remove extra space
    return review


def delete_reporter(text):
    text = text.strip()
    for _ in range(text.count('기자')):
        if text.find('기자') < len(text) // 3:
            text = text[text.find('기자') + 2:]
        else:
            text = text[:text[:text.rfind('기자')].rfind('다') + 1]
    text = text[:text.rfind('다') + 1]
    return text.strip()


def preprocessing(text):
    text = clean_text(text)
    text = delete_reporter(text)
    text = f'[CLS] {text} [SEP]'
    return text