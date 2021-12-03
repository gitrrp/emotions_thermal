import pickle
import streamlit as st
from typing import Tuple
from torch import nn
from torchvision import models, transforms
from PIL import Image

SVM_PATH = 'svm.pkl'

EMOTION_MAP = {0: 'anger',
               1: 'disgust',
               2: 'fear',
               3: 'happy',
               4: 'neutral',
               5: 'sadness',
               6: 'surprise'}


class _Model:
    def __init__(self):
        # initialize both models
        self._vgg = models.vgg16(pretrained=True)
        self._svm = pickle.load(open(SVM_PATH, 'rb'))

        # freeze vgg model
        for param in self._vgg.parameters():
            param.requires_grad = False

        # redefine last layer
        self._vgg.classifier = nn.Sequential(*[self._vgg.classifier[i] for i in range(4)])

        # set model to evaluation
        self._vgg.eval()

    def predict(self, img: Image.Image) -> Tuple[str, bool]:
        """
        Detect emotion from thermal image.

            :param img: image
            :return: emotion, lying or not
        """
        img = img.convert('RGB')
        transformations = transforms.Compose([transforms.Resize(size=(224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        img = transformations(img)[:3, :, :].unsqueeze(0)

        features = self._vgg(img).numpy()
        features.reshape(-1, 1)

        index = self._svm.predict(features)[0]

        emotion = EMOTION_MAP[index]

        return emotion, emotion in {'fear', 'surprise', 'disgust'}


@st.cache
def load_model() -> _Model:
    model = _Model()
    return model
