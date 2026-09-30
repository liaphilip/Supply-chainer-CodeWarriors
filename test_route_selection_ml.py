import networkx as nx

from backend.engine.route_recommender import RouteRecommender


class MockPredictor:
    """
    Controlled predictor for testing whether p85 affects
    the actual Dijkstra route selection.
    """

    def predict_worst_case_delay(
        self,
        origin,
        destination,
        transport_mode,
        nlp_score
    ):
        # Route A has a large p85 delay.
        if destination == "A":
            p85 = 50.0

        # Route B has a small p85 delay.
        elif destination == "B":
            p85 = 10.0

        else:
            p85 = 0.0

        return {
            "p50_delay": p85 * 0.5,
            "p85_delay": p85,
            "p95_delay": p85 * 1.5,
            "confidence_band": "TEST",
            "risk_tier": "TEST",
            "explainability": "Controlled integration test",
            "calibration_reason": "Controlled test profile"
        }


class MockScenarioManager:
    def activate_scenario(self, scenario):
        return None

    def get_active_disruptions(self):
        return {}


class MockResolver:
    def resolve_node_to_entry_point(self, node):
        return {"id": node}


def build_test_graph():
    """
    Creates two possible routes:

        START -> A -> DEST
        START -> B -> DEST

    Route A:
        Normal travel time = 600h
        ML p85 delay = 50h
        Effective = 650h

    Route B:
        Normal travel time = 620h
        ML p85 delay = 10h
        Effective = 630h

    Without ML:
        A would be selected.

    With ML p85:
        B should be selected.
    """

    G = nx.DiGraph()

    # Nodes
    for node in ["START", "A", "B", "DEST"]:
        G.add_node(
            node,
            physical_id=node,
            display_name=node
        )

    # Route A
    G.add_edge(
        "START",
        "A",
        baseline_time=300.0,
        transport_mode="road",
        type="transit",
        cost=100.0
    )

    G.add_edge(
        "A",
        "DEST",
        baseline_time=300.0,
        transport_mode="road",
        type="transit",
        cost=100.0
    )

    # Route B
    G.add_edge(
        "START",
        "B",
        baseline_time=310.0,
        transport_mode="road",
        type="transit",
        cost=100.0
    )

    G.add_edge(
        "B",
        "DEST",
        baseline_time=310.0,
        transport_mode="road",
        type="transit",
        cost=100.0
    )

    return G


def test_ml_p85_changes_route_selection():

    recommender = RouteRecommender.__new__(RouteRecommender)

    recommender.predictor = MockPredictor()
    recommender.scenario_mgr = MockScenarioManager()
    recommender.resolver = MockResolver()
    recommender.unified_graph = build_test_graph()

    result = recommender.recommend(
        source="START",
        destination="DEST",
        transport_preference="any",
        routing_policy="STRICT"
    )

    assert "recommendations" in result
    assert len(result["recommendations"]) > 0

    selected_route = result["recommendations"][0]

    selected_nodes = [
        leg["to"]
        for leg in selected_route["legs"]
    ]

    print("\nML ROUTE SELECTION TEST")
    print("-----------------------------------")
    print("Route A:")
    print("  Normal travel time : 600h")
    print("  ML p85 delay       : +50h")
    print("  Risk-adjusted time : 650h")

    print("\nRoute B:")
    print("  Normal travel time : 620h")
    print("  ML p85 delay       : +10h")
    print("  Risk-adjusted time : 630h")

    print("\nSelected route:")
    print("  ", selected_nodes)

    # The ML-aware router should choose B.
    assert "B" in selected_nodes, (
        f"Expected ML-aware routing to choose Route B, "
        f"but selected {selected_nodes}"
    )

    assert "A" not in selected_nodes, (
        f"Route A was incorrectly selected: {selected_nodes}"
    )

    print("\nPASS: ML p85 prediction changed route selection.")
    print("PASS: Dijkstra is using ML risk in edge weighting.")


if __name__ == "__main__":
    test_ml_p85_changes_route_selection()