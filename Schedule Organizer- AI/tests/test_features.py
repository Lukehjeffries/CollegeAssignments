from features import featurize_task

def test_featurize_empty():
    v = featurize_task("")
    assert len(v) == 2
    assert v[0] == 0

def test_featurize_counts():
    text = "Write a short report"
    v = featurize_task(text)
    assert v[0] == 4
    assert v[1] == len(text)
