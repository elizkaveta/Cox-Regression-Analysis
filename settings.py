RIDGE_LAMBDA = 0.001
TUNING_SEEDS = (100, 101)
FINAL_SEEDS = tuple(range(10))
CHECKPOINT_COUNT = 48
CHECKPOINT_START_UNITS = 128

MENU = {
    "rho_uniform": [
        dict(
            batch_size=batch,
            learning_rate=rate,
            rho=1e-4,
            delta=0.05,
            decay_steps=1000,
            beta_radius=10.0,
            shift_bound=40.0,
        )
        for batch in (128, 256)
        for rate in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    ],
    "batch_lse": [
        dict(outer_batch_size=outer, risk_batch_size=risk, learning_rate=rate)
        for outer, risk, rates in (
            (8, 8, (0.0009, 0.003, 0.009, 0.03, 0.09, 0.3)),
            (16, 64, (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)),
        )
        for rate in rates
    ],
    "minibatch": [
        dict(batch_size=batch, learning_rate=rate)
        for batch in (32, 256)
        for rate in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
    ],
    "bigsurv_normalized": [
        dict(strata_size=20, strata_batch_size=batch, lr_tau=0.5, learning_rate=rate)
        for batch in (1, 16)
        for rate in (0.006, 0.02, 0.06, 0.2, 0.6, 2.0)
    ],
    "cox_cc": [
        dict(case_batch_size=32, n_controls=controls, learning_rate=rate)
        for controls in (1, 8)
        for rate in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
    ],
}


DATASETS = {
    "support2": {
        "d": 22,
        "n_train": 5323,
        "n_val": 1775,
        "n_test": 1775,
        "n_events": 3621,
        "tuning_unit_cap": 3000000,
        "unit_cap": 4500000,
        "data": {
            "file": "support2.csv",
            "duration": "d.time",
            "event": "death",
            "features": [
                "age",
                "sex",
                "race",
                "num.co",
                "diabetes",
                "dementia",
                "ca",
                "meanbp",
                "hrt",
                "resp",
                "temp",
                "wblc",
                "sod",
                "crea",
            ],
            "categories": ["sex", "race", "ca"],
            "kind": "real",
            "url": "https://hbiostat.org/data/repo/support2csv.zip",
        },
        "input_sha256": "b9e56c6e3414ea88f683ee16c52f463fed2cbb80f75c1c18b1cc95ef5e81982e",
        "reference": {
            "loss": 7.957883925107147,
            "gap_resolution": 1.2729918701534522e-13,
        },
    },
    "nwtco": {
        "d": 11,
        "n_train": 2416,
        "n_val": 806,
        "n_test": 806,
        "n_events": 342,
        "tuning_unit_cap": 3000000,
        "unit_cap": 4500000,
        "data": {
            "file": "nwtco.csv",
            "duration": "edrel",
            "event": "rel",
            "features": ["instit", "histol", "stage", "study", "age"],
            "categories": ["instit", "histol", "stage", "study"],
            "kind": "real",
            "url": "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/nwtco.csv",
        },
        "input_sha256": "f7d11bef907e9227ac124203a938ed9d162a156562d06ad12758c98d575b5f73",
        "reference": {
            "loss": 7.3331722550440475,
            "gap_resolution": 1.1842150023318715e-13,
        },
    },
    "syn_n100000_d10_c90_ar095": {
        "d": 10,
        "n_train": 60000,
        "n_val": 20000,
        "n_test": 20000,
        "n_events": 6005,
        "tuning_unit_cap": 10000000,
        "unit_cap": 15000000,
        "data": {
            "kind": "synthetic",
            "n": 100000,
            "d": 10,
            "seed": 1701,
            "censoring": 0.9,
            "signal": 0.5,
            "baseline_shape": 1.5,
            "feature_correlation": 0.95,
        },
        "reference": {
            "loss": 9.528051309718538,
            "gap_resolution": 1.496126075972146e-13,
        },
    },
    "syn_n1000000_d20_c90_ar090": {
        "d": 20,
        "n_train": 600000,
        "n_val": 200000,
        "n_test": 200000,
        "n_events": 60136,
        "tuning_unit_cap": 100000000,
        "unit_cap": 150000000,
        "data": {
            "kind": "synthetic",
            "n": 1000000,
            "d": 20,
            "seed": 1701,
            "censoring": 0.9,
            "signal": 0.5,
            "baseline_shape": 1.5,
            "feature_correlation": 0.9,
        },
        "reference": {
            "loss": 11.80011701284451,
            "gap_resolution": 1.8190060320833643e-13,
        },
    },
    "syn_standard_n100000_d20_c30": {
        "d": 20,
        "n_train": 60000,
        "n_val": 20000,
        "n_test": 20000,
        "n_events": 41921,
        "tuning_unit_cap": 10000000,
        "unit_cap": 15000000,
        "data": {
            "kind": "synthetic",
            "n": 100000,
            "d": 20,
            "seed": 2701,
            "censoring": 0.3,
            "signal": 0.5,
            "baseline_shape": 1.5,
            "feature_correlation": 0.0,
        },
        "reference": {
            "loss": 9.816948025765342,
            "gap_resolution": 1.5371807685607023e-13,
        },
    },
}


REFERENCE_BETA = {
    "support2": (
        0.2133312209831367,
        0.008860071938927723,
        -0.006888878317021573,
        0.05335241153167512,
        -0.07046587060936907,
        0.07208314774295148,
        0.029634979471097993,
        0.01013594003878747,
        0.02856690192516817,
        -0.01119042597174378,
        0.08574182759775402,
        -0.015226106866131694,
        0.015226106866129092,
        0.02470772814196989,
        0.011886297211105491,
        -0.016737173633082775,
        0.0007793398503910371,
        0.01636717491330358,
        -0.012853744057934711,
        0.17202226016617458,
        -0.16080804013449004,
        0.01980343588138443,
    ),
    "nwtco": (
        0.1303168482254677,
        -0.009483776028762134,
        0.009483776028706621,
        -0.23700907335431987,
        0.23700907335431987,
        -0.29018574445305184,
        0.07099578497516053,
        0.10157285009397654,
        0.20271206922139934,
        0.03006075735435356,
        -0.03006075735435356,
    ),
    "syn_n100000_d10_c90_ar095": (
        0.09031326533489846,
        -0.6820938001189235,
        -0.15245169182236726,
        0.3225413986287889,
        0.7237342341526372,
        -0.44272586115996415,
        0.5086869872418822,
        -0.7716484163891641,
        -0.29474469204139814,
        0.3634053757235353,
    ),
    "syn_n1000000_d20_c90_ar090": (
        -0.0161385816442433,
        -0.36925574018598784,
        -0.04448597530095681,
        0.1507130452736212,
        0.365852921789612,
        -0.2533520752386973,
        0.29755083057359355,
        -0.4254563783923001,
        -0.16242513128628006,
        -0.1339564605939225,
        0.07291271736253631,
        0.5344110718230338,
        -0.18773104651604863,
        -0.2484475470956769,
        -0.13437906628721039,
        0.25948218306224513,
        0.14623331663419167,
        -0.18630188976655535,
        0.42279324100370014,
        -0.29075992160700437,
    ),
    "syn_standard_n100000_d20_c30": (
        0.05672467668105448,
        0.07116175534267197,
        0.2034053044739594,
        -0.01832918840631403,
        0.04690032208861103,
        0.03811991700760159,
        -0.1267746341823635,
        0.047605914073620074,
        0.13214874846202013,
        0.01447512578985473,
        0.09446210940343912,
        0.2076017579787593,
        0.13993968463197698,
        -0.14872725417269994,
        -0.06545570832030362,
        -0.10618718304813261,
        -0.05105057510750957,
        -0.14104691978303563,
        0.17916052471047667,
        -0.043655711981328595,
    ),
}
