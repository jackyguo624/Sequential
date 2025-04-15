from lhotse.dataset import K2SpeechRecognitionDataset


class K2SpeechRecognitionDatasetWraper(K2SpeechRecognitionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__getitems__ = self.__getitem__