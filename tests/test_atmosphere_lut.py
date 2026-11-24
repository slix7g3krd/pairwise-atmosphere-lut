import pytest
import graphics.atmosphere_lut as mod

CASES = [
    ("altitude_recalc", (1,)),
    ("lut_smooth", (1,)),
    ("write_fence", (1,)),
    ("turbidity_invalidate", (1,)),
    ("param_hash", (1,)),
    ("sample_determinism", (1,)),
    ("sun_angle", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("precompute", (1,)),
    ("metrics", (1,)),
    ("weather_switch", (1,)),
    ("skybox_bind", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None


# --- behaviour for the six reported defects -------------------------------


def test_altitude_recalc_changes_scattering_coefficients():
    sea = mod.altitude_recalc({"altitude_m": 0.0})
    high = mod.altitude_recalc({"altitude_m": 4000.0})

    assert high["rayleigh"] != sea["rayleigh"]
    assert high["scatter_ratio_vs_sea_level"] < 1.0
    assert high["key"] != sea["key"]
    assert high["checksum"] != sea["checksum"]


def test_lut_smoothing_reduces_sunrise_banding():
    result = mod.lut_smooth({"sun_elevation_deg": 0.0})

    assert result["banding_after"] <= result["banding_before"]
    assert result["banding_reduction"] > 0.0
    assert result["smoothed"] is True


def test_write_fence_publishes_atomically():
    result = mod.write_fence({"altitude_m": 1000.0})

    assert result["atomic_swap"] is True
    assert result["committed_generation"] == result["staged_generation"] + 1
    # Staging must not expose the new table; only commit does.
    assert result["during"]["key"] != result["key"]
    assert result["during"]["published"] == result["before"]["published"]
    assert result["after"]["key"] == result["key"]


def test_turbidity_change_invalidates_old_tables():
    result = mod.turbidity_invalidate({"turbidity": 2.0}, new_turbidity=6.0)

    assert result["changed"] is True
    assert result["key_changed"] is True
    assert result["removed_count"] >= 1
    assert result["stale_served"] is False
    assert result["stale_remaining"] == []


def test_weather_switch_does_not_serve_stale_lut():
    result = mod.weather_switch({"turbidity": 1.5}, turbidity=7.0)

    assert result["old_key"] != result["new_key"]
    assert result["old_key_still_cached"] is False
    assert result["stale_served"] is False


def test_param_hash_is_stable_and_order_independent():
    a = mod.param_hash({"altitude_m": 1500.0, "turbidity": 3.0, "sun_elevation_deg": 5.0})
    b = mod.param_hash({"sun_elevation_deg": 5.0, "turbidity": 3.0, "altitude_m": 1500.0})

    assert a == b
    assert a != mod.param_hash({"altitude_m": 1500.0, "turbidity": 3.5, "sun_elevation_deg": 5.0})
    assert isinstance(a, str) and len(a) == 32


def test_sampling_is_deterministic_at_same_sun_angle():
    result = mod.sample_determinism({"sun_elevation_deg": 12.0})

    assert result["identical"] is True
    assert all(sample == result["samples"][0] for sample in result["samples"])


def test_two_runs_produce_identical_tables():
    first = mod.determinism({"altitude_m": 800.0, "turbidity": 2.5, "sun_elevation_deg": -2.0})
    second = mod.determinism({"altitude_m": 800.0, "turbidity": 2.5, "sun_elevation_deg": -2.0})

    assert first["identical"] is True
    assert first["checksum_first"] == second["checksum_first"]
    assert first["sample_first"] == second["sample_first"]


def test_skybox_bind_reads_published_snapshot():
    result = mod.skybox_bind({"altitude_m": 2500.0})

    assert result["published"] is True
    assert result["stable"] is True
    assert result["key"] == mod.param_hash({"altitude_m": 2500.0})


def test_audit_checks_all_invariants():
    result = mod.audit({"altitude_m": 1200.0, "turbidity": 4.0})

    assert result["ok"] is True
    assert result["failed"] == []


def test_validation_and_fallback_handle_bad_input():
    bad = {"altitude_m": -9999.0, "turbidity": 99.0, "sun_elevation_deg": 400.0}

    report = mod.validation(bad)
    assert report["ok"] is False and report["errors"]

    fixed = mod.fallback(bad)
    assert fixed["used_fallback"] is True
    assert mod.validation(fixed["params"])["ok"] is True
