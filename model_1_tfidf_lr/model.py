from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline(max_features=50_000, ngram_range=(1, 2), C=1.0):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            min_df=2,
        )),
        ("lr", LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
