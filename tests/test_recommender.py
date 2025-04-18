from processing import recommender

def test_recommend_output_structure():
    profiles = [{"keywords": ["trust", "value"], "sentiment": "positive"}]
    result = recommender.recommend(profiles)
    assert "tone" in result
    assert "color_scheme" in result
    assert isinstance(result["keywords"], list)