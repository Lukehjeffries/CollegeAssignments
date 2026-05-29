from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor


class WorkloadEstimator:

    def __init__(self, max_features=2000, ngram_range=(1, 2), random_state=42):
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)),
            ("rf", RandomForestRegressor(random_state=random_state))
        ])

    def fit(self, tasks, hours):
        self.pipe.fit(tasks, hours)

    def predict_hours(self, task):
        pred = self.pipe.predict([task])
        return float(pred[0])

    def get_pipeline(self):
        return self.pipe
