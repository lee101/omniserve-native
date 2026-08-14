from workloads.person_detection import PersonDecision


def test_person_decision_serializes_routing_evidence():
    decision = PersonDecision(
        detected=True,
        sampled_frames=8,
        frames_with_person=6,
        max_confidence=0.91,
        device="cuda:0",
    )
    public = decision.public_dict()
    assert public["detected"] is True
    assert public["sampled_frames"] == 8
    assert public["frames_with_person"] == 6
    assert public["max_confidence"] == 0.91
