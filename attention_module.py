"""
Module nhan dien tu mia mai bang Attention weights.
Dung cho Dashboard Streamlit hoac script danh gia.
Import: from attention_module import load_attention_model, predict_with_attention, render_html_highlight
"""
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return [tf.reduce_sum(output, axis=1),a]

    def get_config(self):
        return super(AttentionLayer, self).get_config()
MAX_LEN = 50


def load_attention_model(model_path="attention_extraction_model.h5", tokenizer_path="tokenizer_final.pkl"):
    """Load model va tokenizer da train san."""
    model = tf.keras.models.load_model(model_path,	custom_objects={'AttentionLayer': AttentionLayer}, compile=False)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict_with_attention(sentence, tokenizer, attention_model, max_len=MAX_LEN):
    """
    Du doan 1 cau va tra ve attention weight cho moi tu.
    Tra ve dict: sentence, prediction (MIA MAI/KHONG MIA MAI),
    confidence, words (list), weights_normalized (list, 0-1).
    """
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    pred, attn = attention_model.predict(padded, verbose=0)
    pred_label = "MIA MAI" if pred[0][0] > 0.5 else "KHONG MIA MAI"
    confidence = float(pred[0][0])

    words = sentence.split()
    n_words = len(words)
    word_weights = attn[0][:n_words]

    if word_weights.max() > 0:
        normalized = (word_weights - word_weights.min()) / (word_weights.max() - word_weights.min() + 1e-9)
    else:
        normalized = word_weights

    return {
        "sentence": sentence,
        "prediction": pred_label,
        "confidence": confidence,
        "words": words,
        "weights_normalized": normalized.tolist(),
    }


def get_top_attention_words(result, top_n=2):
    """Lay top-N tu co attention weight cao nhat trong 1 ket qua predict_with_attention."""
    if len(result["weights_normalized"]) == 0:
        return []
    idx_sorted = np.argsort(result["weights_normalized"])[::-1][:top_n]
    return [result["words"][i] for i in idx_sorted]


def render_html_highlight(result):
    """
    Tra ve doan HTML to mau tu theo attention weight.
    Mau cam dam = trong so cao = tu quan trong cho viec phat hien mia mai.
    Dung truc tiep trong Streamlit: st.markdown(html, unsafe_allow_html=True)
    """
    html = "<div style=\"font-size:16px; line-height:2;\">"
    for word, w in zip(result["words"], result["weights_normalized"]):
        r, g, b = 255, int(255 - w * 140), int(180 - w * 180)
        color = f"rgb({r},{max(g,0)},{max(b,0)})"
        html += f"<span style=\"background-color:{color}; padding:3px 6px; margin:2px; border-radius:4px;\">{word}</span> "
    html += "</div>"
    return html


if __name__ == "__main__":
    model, tokenizer = load_attention_model()
    result = predict_with_attention("san pham tot qua dung xong nhap vien luon", tokenizer, model)
    print("Du doan:", result["prediction"], f"({result['confidence']:.1%})")
    print("Top tu quan trong:", get_top_attention_words(result))
