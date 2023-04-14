from model import buildModel

def test_Model():

    model_score = buildModel()
    assert model_score["score"] >= 0.60