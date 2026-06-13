import matplotlib.pyplot as plt

# Replace with your final decoded feature names and frequencies
features = [
    "predictable",
    "adaptation",
    "surprisingly",
    "social commentary",
    "transformation",
    "great movie",
    "talky",
    "oscar (best picture)",
    "plot",
    "culture clash"
]

frequencies = [10, 9, 7, 7, 6, 5, 3, 3, 2, 2]

plt.figure(figsize=(10,6))

plt.barh(features[::-1], frequencies[::-1])

plt.xlabel("Frequency in Top Attention Features")
plt.ylabel("Metadata Feature")
plt.title("Most Important Metadata Features Based on Attention")

plt.tight_layout()

plt.savefig(
    "outputs/plots/feature_importance_final.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()