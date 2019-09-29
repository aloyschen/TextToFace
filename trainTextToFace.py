import numpy as np
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
            imageDlatent = np.load("./latent_representations/" + imageName.split(".")[0] + ".npy")
            imageDlatents.append(imageDlatent)
    return np.array(textEmbedding), np.array(imageDlatents)

def train():
    model = textTodlatentsModel().build()
    sentenceEmbedding, imageDlatents = loadData("./data/train.txt")
    model.fit(sentenceEmbedding, imageDlatents, batch_size = 5, nb_epoch = 1000)
    model.save("./model/textEmbeddingDlatents.h5")

if __name__ == "__main__":
    train()
