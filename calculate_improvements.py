import pandas as pd

results = {
    "Metric": [
        "Warm RMSE",
        "Warm MAE",
        "Cold RMSE",
        "Cold MAE",
        "Precision@10",
        "Recall@10",
        "HitRatio@10",
        "NDCG@10"
    ],
    "Baseline": [
        1.0249,
        0.8072,
        1.1861,
        0.9891,
        0.681668,
        0.304473,
        0.994478,
        0.857095
    ],
    "Attention": [
        0.9629,
        0.7393,
        0.9628,
        0.7471,
        0.688625,
        0.312250,
        0.995030,
        0.861996
    ]
}

df = pd.DataFrame(results)

improvements = []

for i in range(len(df)):
    baseline = df.loc[i, "Baseline"]
    attention = df.loc[i, "Attention"]

    metric = df.loc[i, "Metric"]

    # Lower is better for RMSE and MAE
    if "RMSE" in metric or "MAE" in metric:
        improvement = ((baseline - attention) / baseline) * 100
    else:
        improvement = ((attention - baseline) / baseline) * 100

    improvements.append(round(improvement, 2))

df["Improvement (%)"] = improvements

print(df)

df.to_csv("outputs/models/improvement_table.csv", index=False)