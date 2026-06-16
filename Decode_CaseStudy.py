import pandas as pd

# Load case study file
case_df = pd.read_csv(
    "outputs/models/cold_start_case_study.csv"
)

# Load tag mapping
tag_map = pd.read_csv("processed_data/tag_mapping.csv")

tag_dict = dict(zip(
    tag_map["tagId"].astype(str),
    tag_map["tag"]
))

# Function to decode metadata feature list
def decode_features(feature_string):

    decoded = []

    for token in str(feature_string).split(","):

        token = token.strip()

        if token in tag_dict:
            decoded.append(tag_dict[token])
        else:
            decoded.append(token)

    return ", ".join(decoded)

case_df["Top Metadata Features"] = (
    case_df["Top Metadata Features"]
    .apply(decode_features)
)

print(case_df.head())

case_df.to_csv(
    "outputs/models/cold_start_case_study_decoded.csv",
    index=False
)

print("Decoded case-study table saved.")
