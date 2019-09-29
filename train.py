import numpy as np
import keras
from model import textTodlatentsModel
from bert_serving.client import BertClient

def loadData(dataPath):
    textEmbedding, imageDlatents = [], []
    bc = BertClient()
    with open(dataPath) as file:
        for line in file.readlines():
            data  = line.strip().split("\t")
            if len(data) != 2:
                continue
            imageName = data[0]
            text = data[1]
            sentenceEmbedding = bc.encode([text])
            textEmbedding.append(sentenceEmbedding)
            imageDlatent = np.load("./data/latent/" + imageName.split(".")[0] + ".npy")
            imageDlatents.append(imageDlatent)
    return np.array(textEmbedding), np.array(imageDlatents)

def train():
    model = textTodlatentsModel().build()
    sentenceEmbedding, imageDlatents = loadData("./data/train.txt")
    model.fit(sentenceEmbedding, imageDlatents, batch_size = 5, epochs = 1000)
    model.save("./model/textEmbeddingDlatents.h5")

def predict(inputText):
    bc = BertClient()
    sentenceEmbedding = bc.encode([inputText])
    model = keras.models.load_model("./model/textEmbeddingDlatents.h5")
    sentenceEmbedding = np.expand_dims(sentenceEmbedding, 0)
    result = model.predict(sentenceEmbedding)
    np.save("./data/predict/test.npy", result)
    print("predict successfully")

if __name__ == "__main__":
    # train()
    predict("一位黄色长发小女孩，大眼睛，正面")
