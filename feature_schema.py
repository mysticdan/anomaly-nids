FEATURE_SET_VERSION = "dual_stage_v1"

VARIANT_FEATURES = {
    "mrmr": [
        "Dst Port",
        "Fwd Header Len",
        "Bwd Pkts/b Avg",
        "Init Fwd Win Byts",
        "Flow IAT Mean",
        "Bwd Blk Rate Avg",
        "Bwd Byts/b Avg",
        "Bwd Pkts/s",
        "Subflow Fwd Byts",
        "Fwd Byts/b Avg",
    ],
    "mutual_information": [
        "Pkt Len Mean",
        "Fwd Seg Size Min",
        "Flow IAT Max",
        "Fwd Pkts/s",
        "Flow IAT Mean",
        "Fwd Header Len",
        "Flow Duration",
        "Bwd Pkts/s",
        "Init Fwd Win Byts",
        "Dst Port",
    ],
    "rf_importance": [
        "Flow Byts/s",
        "Init Bwd Win Byts",
        "Flow IAT Mean",
        "Fwd Pkts/s",
        "Fwd Seg Size Min",
        "Flow IAT Max",
        "Dst Port",
        "Init Fwd Win Byts",
        "Flow Duration",
        "Bwd Pkts/s",
    ],
    "rfe": [
        "Dst Port",
        "Flow Duration",
        "Fwd Pkt Len Mean",
        "Flow IAT Mean",
        "Flow IAT Max",
        "Bwd IAT Std",
        "Fwd Pkts/s",
        "Bwd Pkts/s",
        "Init Fwd Win Byts",
        "Init Bwd Win Byts",
    ],
}

FEATURE_NAME_TO_KEY = {
    "Bwd Blk Rate Avg": "bwd_blk_rate_avg",
    "Bwd Byts/b Avg": "bwd_byts_b_avg",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd Pkts/b Avg": "bwd_pkts_b_avg",
    "Bwd Pkts/s": "bwd_pkts_per_s",
    "Dst Port": "dst_port_feat",
    "Flow Byts/s": "flow_byts_per_s",
    "Flow Duration": "flow_duration",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Mean": "flow_iat_mean",
    "Fwd Byts/b Avg": "fwd_byts_b_avg",
    "Fwd Header Len": "fwd_header_len",
    "Fwd Pkt Len Mean": "fwd_pkt_len_mean",
    "Fwd Pkts/s": "fwd_pkts_per_s",
    "Fwd Seg Size Min": "fwd_seg_size_min",
    "Init Bwd Win Byts": "init_bwd_win_byts",
    "Init Fwd Win Byts": "init_fwd_win_byts",
    "Pkt Len Mean": "pkt_len_mean",
    "Subflow Fwd Byts": "subflow_fwd_byts",
}

ALL_MODEL_FEATURE_NAMES = list(
    dict.fromkeys(feature_name for feature_names in VARIANT_FEATURES.values() for feature_name in feature_names)
)
MODEL_FEATURE_NAMES = list(VARIANT_FEATURES["mrmr"])
FEATURE_KEYS = [FEATURE_NAME_TO_KEY[name] for name in ALL_MODEL_FEATURE_NAMES]
FEATURE_KEY_TO_NAME = {v: k for k, v in FEATURE_NAME_TO_KEY.items()}


def normalize_feature_names(feature_names):
    normalized = []
    for feature_name in feature_names:
        if feature_name in FEATURE_NAME_TO_KEY:
            normalized.append(feature_name)
            continue
        if feature_name in FEATURE_KEY_TO_NAME:
            normalized.append(FEATURE_KEY_TO_NAME[feature_name])
            continue
        raise KeyError(f"Unknown model feature: {feature_name}")
    return normalized


def feature_keys_for_names(feature_names):
    return [FEATURE_NAME_TO_KEY[name] for name in normalize_feature_names(feature_names)]


def ordered_feature_values(features_dict, selected_features=None):
    keys = feature_keys_for_names(selected_features) if selected_features else feature_keys_for_names(MODEL_FEATURE_NAMES)
    return [features_dict[key] for key in keys]


def feature_row_to_vector(row, selected_features=None):
    keys = feature_keys_for_names(selected_features) if selected_features else feature_keys_for_names(MODEL_FEATURE_NAMES)
    return [row[key] for key in keys]


def _self_check():
    assert len(ALL_MODEL_FEATURE_NAMES) == 19
    assert set(ALL_MODEL_FEATURE_NAMES) == set(FEATURE_NAME_TO_KEY)
    assert feature_keys_for_names(["Dst Port", "dst_port_feat"]) == ["dst_port_feat", "dst_port_feat"]
    assert normalize_feature_names(["flow_iat_mean", "Flow Duration"]) == ["Flow IAT Mean", "Flow Duration"]


if __name__ == "__main__":
    _self_check()
    print("feature-schema self-check ok")
