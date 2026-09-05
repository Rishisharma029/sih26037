"""
Tests for Earth Engine Geospatial Road Extraction and Geographic Registry.
"""
import pytest
from scenarios.geospatial.geo_registry import INDIAN_SCENARIO_REGISTRY
from scenarios.geospatial.earth_engine_client import EarthEngineRoadExtractor

def test_indian_scenario_registry_coverage():
    assert len(INDIAN_SCENARIO_REGISTRY) == 5
    assert "SCENARIO_1_VILLAGE" in INDIAN_SCENARIO_REGISTRY
    assert "SCENARIO_2_JUNCTION" in INDIAN_SCENARIO_REGISTRY
    assert "SCENARIO_3_HIGHWAY" in INDIAN_SCENARIO_REGISTRY
    assert "SCENARIO_4_MARKET" in INDIAN_SCENARIO_REGISTRY
    assert "SCENARIO_5_CATTLE" in INDIAN_SCENARIO_REGISTRY

    # Verify real-world coordinates are within Indian boundary
    for key, loc in INDIAN_SCENARIO_REGISTRY.items():
        assert 8.0 <= loc.latitude <= 37.0 # India lat bounds
        assert 68.0 <= loc.longitude <= 97.0 # India lon bounds
        assert loc.carriageway_width_m > 3.0

def test_earth_engine_feature_extractor():
    extractor = EarthEngineRoadExtractor(use_live_api=False)
    features = extractor.get_road_features("SCENARIO_1_VILLAGE")

    assert "elevation_profile" in features
    assert "road_geometry" in features
    assert features["road_geometry"]["paved_width_m"] == 4.2
    assert features["elevation_profile"]["lateral_ditch_drop_m"] < 0.0
