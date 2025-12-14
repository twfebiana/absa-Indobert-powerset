# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_sentiment_counts(df, sentiment_map=None):
    if sentiment_map:
        df['sentiment'] = df['pred_class'].map(sentiment_map)
    else:
        def determine(row):
            vals = [v for k,v in row.items() if k.startswith("aspect_")]
            if any([v==1 for v in vals]):
                return "positive"
            if any([v==-1 for v in vals]):
                return "negative"
            return "neutral"
        df['sentiment'] = df.apply(determine, axis=1)
    counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title("Distribusi Sentimen")
    ax.set_ylabel("Jumlah")
    return fig, df

def plot_aspect_freq(df):
    # aggregate per-aspect positive/negative counts
    aspects = ["kpms","fi","wt","bl"]
    data = {"aspect":[],"positive":[], "negative":[]}
    for a in aspects:
        pos_col = f"aspect_{a}_pos" if f"aspect_{a}_pos" in df.columns else f"aspect_{a}"
        # our inference sets aspect_{a}: 1 pos, -1 neg, 0 none
        pos = (df[f"aspect_{a}"]==1).sum() if f"aspect_{a}" in df.columns else 0
        neg = (df[f"aspect_{a}"]==-1).sum() if f"aspect_{a}" in df.columns else 0
        data["aspect"].append(a)
        data["positive"].append(pos)
        data["negative"].append(neg)
    agg = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(6,4))
    agg_melt = agg.melt(id_vars="aspect", value_vars=["positive","negative"], var_name="polarity", value_name="count")
    sns.barplot(data=agg_melt, x="aspect", y="count", hue="polarity", ax=ax)
    ax.set_title("Frekuensi Aspek (pos vs neg)")
    return fig
