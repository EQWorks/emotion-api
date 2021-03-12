from flask import Flask, request, jsonify
import fasttext


app = Flask(__name__)
model = fasttext.load_model('./emotions.bin')


def emotion_score(model, text):
    labels, scores = model.predict(text.lower())
    return {'emotion': labels[0].split('__label__')[-1], 'score': round(scores[0], 3)}


@app.route('/')
def get_emotion():
    text = request.args.get('t', request.args.get('text', ''))

    if not text:
        return jsonify({'error': 'You need to supply a text'}), 400

    return jsonify({
        'text': text,
        **emotion_score(model, text),
    })


if __name__ == '__main__':
    app.run()
