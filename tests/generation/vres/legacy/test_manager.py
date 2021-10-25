import pytest

from epippy.generation.vres.legacy.manager import *
from epippy.geographics.shapes import get_shapes


def test_get_legacy_capacity_in_countries_wrong_tech():
    with pytest.raises(AssertionError):
        get_legacy_capacity_in_countries('wind', ['BE'])


def test_get_legacy_capacity_in_countries_wrong_tech_plant_type():
    with pytest.raises(ValueError):
        get_legacy_capacity_in_countries('ccgt', ["BE"])
    with pytest.raises(ValueError):
        get_legacy_capacity_in_countries('wind_floating', ["BE"])


def test_get_legacy_capacity_in_countries_wrong_tech_plant_type_warning():
    df = get_legacy_capacity_in_countries('ccgt', ["BE"], False)
    assert df.loc["BE"] == 0
    df = get_legacy_capacity_in_countries('wind_floating', ["BE"], False)
    assert df.loc["BE"] == 0


def test_get_legacy_capacity_in_countries_empty_countries_list():
    with pytest.raises(AssertionError):
        get_legacy_capacity_in_countries('wind_onshore', [])


def test_get_legacy_capacity_in_countries_missing_country():
    for tech in ['wind_onshore', 'wind_offshore', 'pv_utility', 'pv_residential']:
        ds = get_legacy_capacity_in_countries(tech, ['ZZ'])
        assert isinstance(ds, pd.Series)
        assert ds.index.equals(pd.Index(['ZZ']))
        assert ds['ZZ'] == 0


def test_get_legacy_capacity_in_countries_output():
    # Wind onshore
    ds = get_legacy_capacity_in_countries('wind_onshore', ['AL', 'CH'])
    assert isinstance(ds, pd.Series)
    assert ds.index.equals(pd.Index(['AL', 'CH']))
    assert round(ds["AL"], 2) == 0.15
    # Wind offshore
    ds = get_legacy_capacity_in_countries('wind_offshore', ['GR', 'GB'])
    assert isinstance(ds, pd.Series)
    assert ds.index.equals(pd.Index(['GR', 'GB']))
    assert round(ds["GB"], 2) == 39.64
    # PV utility
    ds = get_legacy_capacity_in_countries('pv_utility', ['BE', 'NL', 'LU'])
    assert isinstance(ds, pd.Series)
    assert ds.index.equals(pd.Index(['BE', 'NL', 'LU']))
    assert round(ds["NL"], 2) == 0.67
    # PV residential
    ds = get_legacy_capacity_in_countries('pv_residential', ['ME', 'RO'])
    assert isinstance(ds, pd.Series)
    assert ds.index.equals(pd.Index(['ME', 'RO']))
    assert round(ds["RO"], 2) == 0.04


def test_get_legacy_capacity_in_regions_wrong_tech():
    with pytest.raises(AssertionError):
        get_legacy_capacity_in_regions('wind', pd.Series(index=["BE"]), ["BE"])


def test_get_legacy_capacity_in_regions_wrong_tech_plant_type():
    with pytest.raises(ValueError):
        get_legacy_capacity_in_regions('ccgt', pd.Series(index=["BE"]), ["BE"])
    with pytest.raises(ValueError):
        get_legacy_capacity_in_regions('wind_floating', pd.Series(index=["BE"]), ["BE"])


def test_get_legacy_capacity_in_regions_wrong_tech_plant_type_warning():
    df = get_legacy_capacity_in_regions('ccgt', pd.Series(index=["BE"]), ["BE"], raise_error=False)
    assert df.loc["BE"] == 0
    df = get_legacy_capacity_in_regions('wind_floating', pd.Series(index=["BE"]), ["BE"], raise_error=False)
    assert df.loc["BE"] == 0


def test_get_legacy_capacity_in_regions_missing_countries():
    countries = ["XX", "ZZ"]
    for tech in ["pv_residential", "pv_utility", "wind_onshore", "wind_offshore"]:
        df = get_legacy_capacity_in_regions(tech, pd.Series(index=countries), countries)
        assert df["XX"] == 0.0
        assert df["ZZ"] == 0.0


def test_get_legacy_capacity_in_regions_region_country_mismatch():
    all_shapes = get_shapes(["PT"], which='onshore_offshore')
    # onshore
    onshore_shape = all_shapes.loc[~all_shapes['offshore']]["geometry"]
    for tech in ["pv_residential", "pv_utility", "wind_onshore"]:
        df = get_legacy_capacity_in_regions(tech, onshore_shape, ["FR"])
        assert df["PT"] == 0.0
    # offshore
    offshore_shape = all_shapes.loc[all_shapes['offshore']]["geometry"]
    df = get_legacy_capacity_in_regions('wind_offshore', offshore_shape, ["FR"])
    assert df["PT"] == 0.0


def test_get_legacy_capacity_in_regions_vs_countries():
    all_shapes = get_shapes(["BE", "NL"], which='onshore_offshore')
    # onshore
    onshore_shapes = all_shapes.loc[~all_shapes['offshore']]["geometry"]
    for tech in ["pv_residential", "pv_utility", "wind_onshore"]:
        df1 = get_legacy_capacity_in_regions(tech, onshore_shapes, ["BE", "NL"])
        df2 = get_legacy_capacity_in_countries(tech, ["BE", "NL"])
        assert abs(df1["BE"]-df2["BE"])/max(df1["BE"], df2["BE"]) < 0.1
        assert abs(df1["NL"]-df2["NL"])/max(df1["NL"], df2["NL"]) < 0.1

    # offshore
    offshore_shapes = all_shapes.loc[all_shapes['offshore']]["geometry"]
    df1 = get_legacy_capacity_in_regions('wind_offshore', offshore_shapes, ["BE", "NL"])
    df2 = get_legacy_capacity_in_countries('wind_offshore', ["BE", "NL"])
    assert abs(df1["BE"] - df2["BE"]) / max(df1["BE"], df2["BE"]) < 0.1
    assert abs(df1["NL"] - df2["NL"]) / max(df1["NL"], df2["NL"]) < 0.1
