import speech_recognition as sr
import logging
from audiorecognizer import Extract_embeddings
from scipy.spatial.distance import cosine
import numpy as np


class Speaker_identification:
    def __init__(self, threshold=0.5):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
        self.data_audio = {}
        self.id = 0
        self.threshold = threshold
        self.recognizer = sr.Recognizer()
        self.processed_audio = Extract_embeddings()

    def verify_id(self, embedding):
        embedding = embedding[0]
        if not self.data_audio:
            self.data_audio[self.id] = [embedding]
            return
        max_distance = float("inf")
        current_id = None
        for class_id, embeddings in self.data_audio.items():
            distance = self.calculate_distance(embedding, embeddings)
            if distance < max_distance:
                max_distance = distance
                current_id = class_id
        if max_distance < self.threshold:
            self.data_audio[current_id].append(embedding)
            logging.info(f"Assigned id: {current_id}")
        else:
            self.id += 1
            self.data_audio[self.id] = [embedding]
            logging.info(f"New id assigned: {self.id}")

    def calculate_distance(self, embedding, stored_embeddings):
        if len(stored_embeddings) > 1:
            mean_embeddings = np.mean(stored_embeddings, axis=0)
            return cosine(embedding, mean_embeddings)
        return cosine(embedding, stored_embeddings[0])

    def listen_command(self):
        with sr.Microphone() as source:
            while True:
                logging.info("Speak")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio_data = self.recognizer.listen(source)
                try:
                    extracted_embedding = self.processed_audio.collect_embedding(
                        audio_data.get_wav_data()
                    )
                    self.verify_id(extracted_embedding)
                except sr.UnknownValueError:
                    logging.info("Unable to understand audio.")


if __name__ == "__main__":
    sp = Speaker_identification(threshold=0.5)
    sp.listen_command()
