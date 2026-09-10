from fastapi.testclient import TestClient

from analysis import COEFFICIENTS, DEFAULT_POINTS, DEFAULT_TOLERANCE, run_analysis
from app import app
from charts import z_plane_chart
from polynomial import evaluate_polynomial


client = TestClient(app)


def test_complete_activity_audit():
    data = run_analysis(*DEFAULT_POINTS, DEFAULT_TOLERANCE, 50)
    roots = data["roots"]
    assert len(roots) == 4
    assert data["first"].root == roots[0]
    assert all(abs(evaluate_polynomial(COEFFICIENTS, root)) < 1e-8 for root in roots)
    expected_stability = all(abs(root) < 1 for root in roots)
    assert data["stable"] is expected_stability
    assert expected_stability is True


def test_z_plane_uses_exact_calculated_roots():
    data = run_analysis(*DEFAULT_POINTS, DEFAULT_TOLERANCE, 50)
    roots = data["roots"]
    figure = z_plane_chart(roots, data["bound"])
    root_trace = figure.data[-1]
    assert list(root_trace.x) == [root.real for root in roots]
    assert list(root_trace.y) == [root.imag for root in roots]


def test_iteration_error_series_is_present_and_consistent():
    data = run_analysis(*DEFAULT_POINTS, DEFAULT_TOLERANCE, 50)
    rows = data["first"].iterations
    assert rows
    for row in rows:
        expected = abs((row["z_new"] - row["z2"]) / row["z_new"])
        assert abs(row["ea"] - expected) < 1e-15


def test_vercel_fastapi_entrypoint_and_analysis_endpoint():
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json() == {"status": "ok"}
    response = client.post("/api/analyze", json={
        "z0": 0, "z1": 0.5, "z2": 1, "tolerance": 1e-5, "max_iterations": 50,
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["stable"] is True
    assert len(payload["roots"]) == 4
    assert payload["first"]["iteration_count"] == 6
