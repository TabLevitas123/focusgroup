from processing import profile_builder as pb

def test_empty_utterance_list_returns_empty_profiles():
    profiles = pb.build_profiles([])
    assert isinstance(profiles, list)
    assert len(profiles) == 0

def test_utterances_with_weird_tokens():
    utterances = [pb.Utterance(0, 1, "A", "love!!!...#&^@ this 123product")]
    profiles = pb.build_profiles(utterances)
    assert "love!!!...#&^@" not in profiles[0].keywords