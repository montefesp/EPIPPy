import pytest

from iepy.generation.vres.potentials.glaes import *

# All these tests were run with a pixelRes set to 1000
def check_correctness(expected, actual):
    assert np.abs(actual - expected) / expected < 0.05


def test_get_glaes_prior_defaults_empty_config_list():
    with pytest.raises(AssertionError):
        get_glaes_prior_defaults([])


def test_get_glaes_prior_defaults_wrong_exclusion_file():
    with pytest.raises(AssertionError):
        get_glaes_prior_defaults(["wrong"])


def test_get_glaes_prior_defaults_wrong_subconfig():
    with pytest.raises(AssertionError):
        get_glaes_prior_defaults(["holtinger", "wrong"])


def test_get_glaes_prior_defaults_absent_prior():
    with pytest.raises(AssertionError):
        get_glaes_prior_defaults(["holtinger", "wind_onshore", "min"], ["wrong"])


def test_get_glaes_prior_defaults_all_priors():
    dct = get_glaes_prior_defaults(["holtinger", "wind_onshore", "min"])
    assert len(dct.keys()) == 18


def test_get_glaes_prior_defaults():
    priors = ["airport_proximity", "river_proximity"]
    dct = get_glaes_prior_defaults(["holtinger", "wind_onshore", "min"], priors)
    assert len(dct.keys()) == len(priors)
    assert all([p in dct for p in priors])


def test_compute_land_availability_missing_globals():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    with pytest.raises(NameError):
        compute_land_availability(onshore_shape)


def test_compute_land_availability_empty_filters():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    init_land_availability_globals({})
    availability = compute_land_availability(onshore_shape)
    check_correctness(30683.0, availability)

    offshore_shape = get_shapes(["BE"], "offshore").loc["BE", "geometry"]
    init_land_availability_globals({})
    availability = compute_land_availability(offshore_shape)
    check_correctness(3454.0, availability)


def test_compute_land_availability_esm():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    filters = {'esm': True}
    init_land_availability_globals(filters)
    availability = compute_land_availability(onshore_shape)
    check_correctness(11542.83, availability)


def test_compute_land_availability_glaes_priors():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    filters = {'glaes_priors': {'settlement_proximity': (None, 1000)}}
    init_land_availability_globals(filters)
    availability = compute_land_availability(onshore_shape)
    check_correctness(6122.68, availability)

    offshore_shape = get_shapes(["BE"], "offshore").loc["BE", "geometry"]
    filters = {'glaes_priors': {'shore_proximity': [(None, 20e3), (370e3, None)]}}
    init_land_availability_globals(filters)
    availability = compute_land_availability(offshore_shape)
    check_correctness(2125.0, availability)


def test_compute_land_availability_natura():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    filters = {'natura': 1}
    init_land_availability_globals(filters)
    availability = compute_land_availability(onshore_shape)
    check_correctness(26821.79, availability)

    offshore_shape = get_shapes(["BE"], "offshore").loc["BE", "geometry"]
    filters = {'natura': 1}
    init_land_availability_globals(filters)
    availability = compute_land_availability(offshore_shape)
    check_correctness(2197.84, availability)


def test_compute_land_availability_gebco():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    filters = {'altitude_threshold': 300}
    init_land_availability_globals(filters)
    availability = compute_land_availability(onshore_shape)
    check_correctness(24715.12, availability)

    offshore_shape = get_shapes(["BE"], "offshore").loc["BE", "geometry"]
    filters = {'depth_thresholds': {'low': -50, 'high': -10}}
    init_land_availability_globals(filters)
    availability = compute_land_availability(offshore_shape)
    check_correctness(2828.41, availability)


def test_compute_land_availability_emodnet():
    offshore_shape = get_shapes(["BE"], "offshore").loc["BE", "geometry"]
    filters = {'cables': 500}
    init_land_availability_globals(filters)
    availability = compute_land_availability(offshore_shape)
    check_correctness(3115.0, availability)

    filters = {'pipelines': 500}
    init_land_availability_globals(filters)
    availability = compute_land_availability(offshore_shape)
    check_correctness(3287.0, availability)

    filters = {'shipping': (100, None)}
    init_land_availability_globals(filters)
    availability = compute_land_availability(offshore_shape)
    check_correctness(1661.0, availability)


def test_get_land_availability_for_shapes_empty_list_of_shapes():
    with pytest.raises(AssertionError):
        get_land_availability_for_shapes([], {})


def test_get_land_availability_for_shapes_mp_vs_non_mp():
    onshore_shapes = get_shapes(["BE", "NL"], "onshore")["geometry"]
    filters = {'glaes_priors': {'settlement_proximity': (None, 1000)}}
    availabilities_mp = get_land_availability_for_shapes(onshore_shapes, filters)
    availabilities_non_mp = get_land_availability_for_shapes(onshore_shapes, filters, 1)
    assert len(availabilities_mp) == 2
    assert all([availabilities_mp[i] == availabilities_non_mp[i] for i in range(2)])


def test_get_capacity_potential_for_shapes():
    onshore_shapes = get_shapes(["BE", "NL"], "onshore")["geometry"]
    filters = {'glaes_priors': {'settlement_proximity': (None, 1000)}}
    power_density = 10
    capacities = get_capacity_potential_for_shapes(onshore_shapes, filters, power_density)
    assert len(capacities) == 2
    check_correctness(61.2268, capacities[0])
    check_correctness(198.8756, capacities[1])

    offshore_shapes = get_shapes(["BE", "NL"], "offshore")["geometry"]
    filters = {'natura': 1}
    power_density = 15
    capacities = get_capacity_potential_for_shapes(offshore_shapes, filters, power_density)
    assert len(capacities) == 2
    check_correctness(32.9676, capacities[0])
    check_correctness(715.9119, capacities[1])


def test_get_capacity_potential_per_country():
    filters = {'glaes_priors': {'settlement_proximity': (None, 1000)}}
    power_density = 10
    capacities_ds = get_capacity_potential_per_country(["BE", "NL"], True, filters, power_density)
    assert isinstance(capacities_ds, pd.Series)
    assert len(capacities_ds) == 2
    check_correctness(61.2268, capacities_ds["BE"])
    check_correctness(198.8756, capacities_ds["NL"])

    filters = {'natura': 1}
    power_density = 15
    capacities_ds = get_capacity_potential_per_country(["BE", "NL"], False, filters, power_density)
    assert isinstance(capacities_ds, pd.Series)
    assert len(capacities_ds) == 2
    check_correctness(32.9676, capacities_ds["BE"])
    check_correctness(715.9119, capacities_ds["NL"])
