import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Global Style Config ---
sns.set_theme(
    style="whitegrid",
    palette="viridis",
    context="talk"
)

plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (10, 5)

IMAGE_DIR =  "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def load_titanic():
    """
    Load the Titanic dataset directly from seaborn's built-in datasets.
    No file download needed. 
    """

    df = sns.load_dataset("titanic")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def basic_info(df):
    """Print shape, dtypes, missing values, and basic statistics."""
    print("\n=== BASIC INFO ===")
    print(f"Shape       : {df.shape}")
    print(f"Columns     : {list(df.columns)}")
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    })
    print(missing_df[missing_df["Missing Count"] > 0])
    print("\n--- Descriptive Stats ---")
    print(df.describe().round(2))


def survival_rate_by_feature(df, feature):
    """
    Print and return survival rate grouped by a categorical feature.
    Example: survival_rate_by_feature(df, 'sex')
    """

    result = df.groupby(feature)["survived"].agg(["mean", "count"])
    result.columns = ["Survival Rate", "Count"]
    result["Survival Rate"] = (result["Survival Rate"] * 100).round(1)
    print(f"\n--- Survival Rate by {feature.upper()} ---")
    print(result)
    return result

def plot_survival_by_class(df):
    """Bar plot: survival rate per passenger class."""

    fig, ax = plt.subplots()
    class_survival = df.groupby("pclass")["survived"].mean() * 100
    class_colors = ["#2563EB", "#10B981", "#F59E0B"]
    class_survival.plot(
    kind="bar",
    color=class_colors,
    ax=ax,
    width=0.5
)
    ax.set_title("Survival Rate by Passenger Class", fontsize=15, fontweight="bold")
    ax.set_xlabel("Passenger Class (1=First, 3=Third)")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)

    for i, v in enumerate(class_survival):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "survival_by_class.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

def plot_survival_by_sex(df):
    """Bar plot: survival rate by sex."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sex_survival = df.groupby("sex")["survived"].mean() * 100
    sex_colors = ["#EC4899", "#3B82F6"] 
    sex_survival.plot(
    kind="bar",
    color=sex_colors,
    ax=ax,
    width=0.4
)
    ax.set_title("Survival Rate by Sex", fontsize=15, fontweight="bold")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Survival Rate(%)")
    ax.set_ylim(0, 100)
    for i, v in enumerate(sex_survival):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "survival_by_sex.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

def plot_age_distribution(df):
    """Histogram of age distribution with survival overlay."""
    fig, ax = plt.subplots()
    survived = df[df["survived"] == 1]["age"].dropna()
    died = df[df["survived"] == 0]["age"].dropna()
    ax.hist(died, bins=30, alpha=0.6, color="#7d180c", label="Did not survive")
    ax.hist(survived, bins=30, alpha=0.6, color="#0B5128", label="Survived")
    ax.set_title("Age Distribution by Survival", fontsize=15, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Passengers")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "age_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

def plot_heatmap(df):
    """Correlation heatmap of numeric features."""

    fig, ax = plt.subplots(figsize=(8, 5))
    numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[numeric_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        linewidths=0.5
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

def plot_fare_by_class(df):
    """Box plot: fare distribution per passenger class."""
    fig, ax = plt.subplots()
    sns.boxplot(
    data=df,
    x="pclass",
    y="fare",
    color="#60A5FA",
    ax=ax
)
    ax.set_title("Fare Distribution by Passenger Class", fontsize=15, fontweight="bold")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Fare (£)")
    ax.set_ylim(0, 300)
    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "fare_by_class.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

def run_full_analysis():
    """Run the complete EDA pipeline and save all plots."""
    print("Starting Titanic EDA...\n")
    df = load_titanic()
    basic_info(df)
    survival_rate_by_feature(df, "sex")
    survival_rate_by_feature(df, "pclass")
    survival_rate_by_feature(df, "embark_town")
    plot_survival_by_class(df)
    plot_survival_by_sex(df)
    plot_age_distribution(df)
    plot_heatmap(df)
    plot_fare_by_class(df)
    print("\nEDA complete. All plots saved to images/")

if __name__ == "__main__":
    run_full_analysis()