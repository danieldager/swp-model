"""Surprisal control for the delta-norm-by-position analysis.

Phonemes late in a word are more predictable than phonemes early in it, so a ||delta||
that falls with position may be a predictability effect wearing a position costume.
This module puts a number on that, for ||delta_state|| (Dz).

Two estimates of predictability, both over the training lexicon:

    s_bigram    -log2 P(p_t | previous phoneme), add-k smoothed
    s_trigram   the same with two phonemes of context (computed, not plotted)
    s_cohort    -log2 P(p_t | full prefix) over the frequency mass of the words still
                matching that prefix; backs off to the bigram when the cohort is empty

The trained model's <EOS> is a real token, so a word-final phoneme already competes
against "the word could have stopped here" without any extra boundary symbol.

    build_table      one row per phoneme token: norms, position, all three surprisals
    regression_table norm ~ position / surprisal / both -- standardized, cluster-robust
    compare_plot     figure 1 -- norm and surprisal against position, and against each other
    regression_plot  figure 2 -- R^2 partition and standardized betas
    stratified_plot  figure 3 -- norm by position within surprisal strata, and residualized

Position 0 is dropped throughout: ``delta[0]`` is ``h[0]`` rather than a difference (see
``states_extract``), and a word-initial phoneme has no context to be surprising in.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr

from intervention.paths import STATES_DIR, get_train_dataset
from intervention.plotting.style import COLORS, INK, STRATA_COLORS, SURPRISAL_COLOR, paper_style
from intervention.state_analysis.states_extract import StatesDataset

BOS = "<S>"
TARGETS = ["delta_h", "delta_c", "delta_state"]   # all computed; only NORM is plotted
NORM = "norm_delta_state"                         # the one the report shows, ||Dz||
SUR_COLS = ["s_bigram", "s_trigram", "s_cohort"]  # all computed
FIGURE_SURS = ["s_bigram", "s_cohort"]            # the two the figures use
MAX_POS = 12   # past this the per-position n is in the double digits and the curve is noise
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots" / "surprisal"


# --------------------------------------------------------------------------- #
# The probability models. All expose prob(prefix, phoneme) so the caller can swap
# one for another, or chain them as model -> backoff.
# --------------------------------------------------------------------------- #
def fit_ngram(seqs, weights, n: int = 2, k: float = 0.1):
    """Option A: add-k smoothed P(phoneme | previous n-1 phonemes)."""
    ctx, joint, vocab = Counter(), Counter(), set()
    for seq, w in zip(seqs, weights):
        vocab.update(seq)
        padded = [BOS] * (n - 1) + list(seq)
        for i in range(n - 1, len(padded)):
            c = tuple(padded[i - n + 1 : i])
            ctx[c] += w
            joint[(c, padded[i])] += w
    v_size = len(vocab)

    def prob(prefix, phoneme):
        padded = [BOS] * (n - 1) + list(prefix)
        c = tuple(padded[len(padded) - n + 1 :]) if n > 1 else ()
        return (joint[(c, phoneme)] + k) / (ctx[c] + k * v_size)

    return prob


def fit_cohort(seqs, weights):
    """Option B: P(phoneme | prefix) as the frequency share of the surviving cohort.

    Returns ``None`` when the prefix matches no word, or when no word continues it
    with this phoneme -- the caller decides how to back off.
    """
    nxt: dict[tuple, Counter] = defaultdict(Counter)
    for seq, w in zip(seqs, weights):
        for i, phoneme in enumerate(seq):
            nxt[tuple(seq[:i])][phoneme] += w
    totals = {prefix: sum(c.values()) for prefix, c in nxt.items()}

    def prob(prefix, phoneme):
        counts = nxt.get(tuple(prefix))
        if counts is None or not counts[phoneme]:
            return None
        return counts[phoneme] / totals[tuple(prefix)]

    return prob


def surprisal(seqs, prob, backoff=None) -> np.ndarray:
    """Per-token -log2 P, flattened in sequence order to match the metadata rows."""
    out = []
    for seq in seqs:
        for i, phoneme in enumerate(seq):
            p = prob(seq[:i], phoneme)
            if p is None and backoff is not None:
                p = backoff(seq[:i], phoneme)
            out.append(-np.log2(p) if p else np.nan)
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# Table: one row per phoneme token, with its norms and all three surprisals
# --------------------------------------------------------------------------- #
def build_table(states_path: Path = STATES_DIR / "train_states") -> pd.DataFrame:
    """Load the states, attach delta norms, and score every token under A and B."""
    ds = StatesDataset.load(str(states_path))
    df = ds.metadata.copy()
    for target in TARGETS:
        df[f"norm_{target}"] = np.linalg.norm(ds.get_embeddings(target), axis=1)

    df = df.sort_values(["seq_id", "position"]).reset_index(drop=True)
    seqs = df.groupby("seq_id", sort=True)["phoneme"].apply(list)

    # Log frequency as the word weight: raw token counts let "the" swamp the cohort.
    zipf = get_train_dataset()["Zipf_Frequency"].reindex(seqs.index)
    weights = zipf.fillna(zipf.median()).clip(lower=0.1).to_numpy()

    bigram = fit_ngram(seqs, weights, n=2)
    df["s_bigram"] = surprisal(seqs, bigram)
    df["s_trigram"] = surprisal(seqs, fit_ngram(seqs, weights, n=3))
    df["s_cohort"] = surprisal(seqs, fit_cohort(seqs, weights), backoff=bigram)

    df["word_len"] = df.groupby("seq_id")["position"].transform("max")
    return df[df["position"] > 0].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Regression: standardized, cluster-robust
# --------------------------------------------------------------------------- #
MODELS = {"Position": ["position"],
          "Surprisal": ["surprisal"],
          "Position + surprisal": ["position", "surprisal"]}


def regression_table(tokens: pd.DataFrame, sur: str, cluster: str = "seq_id") -> pd.DataFrame:
    """Fit each model in ``MODELS``; one row per (model, term).

    Predictors and ||delta|| are z-scored, so a beta reads as "SD of ||delta|| per SD of
    the predictor" and position and surprisal can be compared on one axis.

    Standard errors are clustered on the word: the ~187k tokens come from 30k words and
    tokens within a word share both a trajectory and a lexical prefix, so the ordinary
    errors are far too small.

    ``unique_r2`` is the drop in R^2 from removing that term from its own model -- the
    variance only that predictor explains, which is what "controlling for surprisal"
    ultimately comes down to.
    """
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    data = pd.DataFrame({"norm": z(tokens[NORM]), "position": z(tokens["position"]),
                         "surprisal": z(tokens[sur]), "cluster": tokens[cluster].values})

    def fit(terms):
        return smf.ols("norm ~ " + " + ".join(terms), data=data).fit(
            cov_type="cluster", cov_kwds={"groups": data["cluster"]})

    rows = []
    for name, terms in MODELS.items():
        res = fit(terms)
        for term in terms:
            reduced = [other for other in terms if other != term]
            unique = res.rsquared - (fit(reduced).rsquared if reduced else 0.0)
            beta, err = res.params[term], res.bse[term]
            rows.append({"surprisal": sur, "model": name, "term": term,
                         "beta": beta, "se": err, "ci_low": beta - 1.96 * err,
                         "ci_high": beta + 1.96 * err, "t": beta / err,
                         "r2": res.rsquared, "unique_r2": unique, "n": len(data)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figure 1: is position confounded with surprisal at all?
# --------------------------------------------------------------------------- #
def _curve(frame, x, y, min_count=1):
    """Mean, SEM and n of `y` at each level of `x`."""
    curve = frame.groupby(x)[y].agg(["mean", "sem", "count"]).reset_index()
    return curve[curve["count"] >= min_count]


def _line(axis, curve, color, label=None, band=True):
    """The house line: SEM band, solid stroke, open marker."""
    if band:
        axis.fill_between(curve.iloc[:, 0], curve["mean"] - curve["sem"],
                          curve["mean"] + curve["sem"], color=color, alpha=0.18, lw=0)
    axis.plot(curve.iloc[:, 0], curve["mean"], color=color, lw=2.0, marker="o", ms=7,
              mfc="white", mec=color, mew=2, zorder=3, label=label)


def compare_plot(tokens: pd.DataFrame, sur: str, min_count: int = 5, n_bins: int = 8):
    """Left: ||Dz|| and surprisal against position on twin axes -- if the two curves
    fall together, position and surprisal are confounded. Right: ||Dz|| against
    surprisal directly, token density behind binned means.
    """
    paper_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6))

    norm_curve = _curve(tokens, "position", NORM, min_count)
    sur_curve = _curve(tokens, "position", sur, min_count)
    twin = left.twinx()
    for axis, curve, color, label in ((left, norm_curve, COLORS["All"], r"$\|\Delta z\|$"),
                                      (twin, sur_curve, SURPRISAL_COLOR, "Surprisal (bits)")):
        _line(axis, curve, color)
        axis.set_ylabel(label, color=color, labelpad=8)
        axis.tick_params(axis="y", colors=color)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    left.set_xlabel("Phoneme position in word", labelpad=8)
    left.set_xticks(sorted(norm_curve.position))
    left.grid(axis="y", alpha=0.25, lw=0.6)
    twin.grid(False)
    sns.despine(ax=left, right=False)
    sns.despine(ax=twin, right=False)

    # 187k tokens: a scatter would be a solid block, so show density and bin the means.
    right.hexbin(tokens[sur], tokens[NORM], gridsize=45, cmap="Blues", bins="log",
                 mincnt=1, linewidths=0, zorder=2)
    edges = np.unique(np.quantile(tokens[sur], np.linspace(0, 1, n_bins + 1)))
    binned = tokens.assign(bin=pd.cut(tokens[sur], edges, include_lowest=True))
    grouped = binned.groupby("bin", observed=True)
    right.errorbar(grouped[sur].mean(), grouped[NORM].mean(), grouped[NORM].sem(),
                   color=SURPRISAL_COLOR, lw=2.0, marker="o", ms=7, mfc="white",
                   mec=SURPRISAL_COLOR, mew=2, capsize=3, zorder=5)

    r_pearson, _ = pearsonr(tokens[sur], tokens[NORM])
    rho, _ = spearmanr(tokens[sur], tokens[NORM])
    right.set_title(fr"r = {r_pearson:.2f}    $\rho$ = {rho:.2f}    n = {len(tokens):,}",
                    fontsize=9, pad=6)
    right.set_xlabel(f"{sur.replace('s_', '')} surprisal (bits)", labelpad=8)
    right.set_ylabel(r"$\|\Delta z\|$", labelpad=8)
    right.grid(axis="y", alpha=0.25, lw=0.6)
    sns.despine(ax=right)
    figure.tight_layout()

    stats = pd.DataFrame([
        {"pair": f"norm ~ {sur}", "r": r_pearson, "rho": rho},
        {"pair": f"{sur} ~ position", "r": pearsonr(tokens.position, tokens[sur])[0],
         "rho": spearmanr(tokens.position, tokens[sur])[0]},
        {"pair": "norm ~ position", "r": pearsonr(tokens.position, tokens[NORM])[0],
         "rho": spearmanr(tokens.position, tokens[NORM])[0]}]).assign(n=len(tokens))
    return figure, stats


# --------------------------------------------------------------------------- #
# Figure 2: what each predictor is worth
# --------------------------------------------------------------------------- #
def regression_plot(table: pd.DataFrame):
    """Left: R^2 per model, the full model's bar split into what each predictor explains
    alone and what the two share. Right: standardized betas with 95% cluster-robust
    intervals, one group per model.
    """
    paper_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8),
                                         gridspec_kw={"width_ratios": [1, 1.15]})
    order = list(MODELS)
    term_colors = {"position": COLORS["All"], "surprisal": SURPRISAL_COLOR}
    by_model = table.groupby("model")

    r_squared = [by_model.get_group(name).r2.iloc[0] for name in order]
    left.bar(range(len(order)), r_squared, width=0.6, color="#C7CDD4", zorder=3)

    full = by_model.get_group(order[-1])
    unique = {row.term: row.unique_r2 for row in full.itertuples()}
    shared = max(full.r2.iloc[0] - sum(unique.values()), 0.0)
    bottom = 0.0
    for term, share in list(unique.items()) + [("shared", shared)]:
        left.bar(len(order) - 1, share, bottom=bottom, width=0.6, zorder=4,
                 color=term_colors.get(term, "#8C8C8C"),
                 label="shared" if term == "shared" else f"{term} only")
        bottom += share
    for index, value in enumerate(r_squared):
        left.annotate(f"{value:.3f}", (index, value), ha="center", va="bottom",
                      fontsize=9, xytext=(0, 3), textcoords="offset points")
    left.set_xticks(range(len(order)))
    left.set_xticklabels([name.replace(" + ", "\n+ ") for name in order])
    left.set_ylabel(r"$R^2$", labelpad=8)
    left.set_ylim(0, max(r_squared) * 1.35)
    left.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.2)
    left.grid(axis="y", alpha=0.25, lw=0.6)
    sns.despine(ax=left)

    offsets = {"position": -0.16, "surprisal": 0.16}
    for row in table.itertuples():
        centre = order.index(row.model) + (offsets[row.term] if len(MODELS[row.model]) > 1 else 0)
        color = term_colors[row.term]
        right.errorbar(centre, row.beta, yerr=1.96 * row.se, color=color, lw=0,
                       elinewidth=1.6, capsize=4, marker="o", ms=8, mfc="white",
                       mec=color, mew=2, zorder=4)
        right.annotate(f"{row.beta:+.2f}", (centre, row.beta), fontsize=8, ha="left",
                       va="center", xytext=(11, 0), textcoords="offset points", color=color)
    right.axhline(0, color="#5A5A5A", ls="--", lw=1.0, zorder=2)
    right.set_xticks(range(len(order)))
    right.set_xticklabels([name.replace(" + ", "\n+ ") for name in order])
    right.set_xlim(-0.5, len(order) - 0.35)
    right.set_ylabel(r"Standardized $\beta$ on $\|\Delta z\|$", labelpad=8)
    handles = [plt.Line2D([], [], color=color, marker="o", ms=7, mfc="white", mew=2,
                          lw=0, label=term.capitalize()) for term, color in term_colors.items()]
    right.legend(handles=handles, frameon=False, loc="best", handlelength=1.2)
    right.grid(axis="y", alpha=0.25, lw=0.6)
    sns.despine(ax=right)
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# Figure 3: does the decline survive the control?
# --------------------------------------------------------------------------- #
def add_strata(tokens: pd.DataFrame, sur: str, n_strata: int = 3) -> pd.DataFrame:
    """Label each token with its surprisal stratum, cut over the pooled tokens.

    Ranked, not raw: cohort surprisal is exactly 0 for every post-uniqueness-point
    token and those ties collapse the quantile edges.
    """
    labels = {2: ["Low", "High"], 3: ["Low", "Mid", "High"]}.get(
        n_strata, [f"Q{i + 1}" for i in range(n_strata)])
    strata = pd.qcut(tokens[sur].rank(method="first"), n_strata, labels=labels)
    return tokens.assign(stratum=pd.Categorical(strata, categories=labels, ordered=True))


def residualize(tokens: pd.DataFrame, sur: str, n_bins: int = 20) -> np.ndarray:
    """||Dz|| with the effect of surprisal removed, re-centred on the grand mean.

    Binned rather than linear: cohort surprisal is sharply non-linear against ||Dz||
    (it floors at 0 bits past the uniqueness point), so a straight-line fit would leave
    structure behind and the residual would overstate what position explains. Each token
    loses the mean of its own surprisal bin, which removes the effect whatever its shape.
    """
    bins = pd.qcut(tokens[sur].rank(method="first"), n_bins, labels=False)
    values = tokens[NORM]
    return (values - values.groupby(bins).transform("mean") + values.mean()).to_numpy()


def stratified_plot(tokens: pd.DataFrame, sur: str, n_strata: int = 3,
                    min_count: int = 4, by_type: bool = False):
    """Left: ||Dz|| by position, one line per surprisal stratum. If every line still
    slopes down, position survives the control; if they flatten, it does not.
    Right: ||Dz|| with surprisal regressed out, the same control as a single curve
    (split by phoneme type when ``by_type``).
    """
    tokens = add_strata(tokens, sur, n_strata)
    tokens = tokens.assign(residual=residualize(tokens, sur))
    strata = list(tokens.stratum.cat.categories)
    colors = dict(zip(strata, STRATA_COLORS if len(strata) <= 3
                      else sns.color_palette("Blues", len(strata))))

    paper_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    stats, positions = [], set()

    for stratum in strata:
        rows = tokens[tokens.stratum == stratum]
        curve = _curve(rows, "position", NORM, min_count)
        if curve.empty:
            continue
        positions.update(curve.position)
        rows = rows[rows.position.isin(curve.position)]     # statistics match the plot
        _line(left, curve, colors[stratum],
              f"{stratum} ({rows[sur].median():.1f} bits)")
        rho, p_value = spearmanr(rows.position, rows[NORM])
        stats.append({"panel": "stratum", "group": str(stratum), "n": len(rows),
                      "median_surprisal": rows[sur].median(), "rho": rho, "p": p_value})

    for group in (["Consonant", "Vowel"] if by_type else ["All"]):
        rows = tokens if group == "All" else tokens[tokens.type == group]
        curve = _curve(rows, "position", "residual", min_count)
        positions.update(curve.position)
        rows = rows[rows.position.isin(curve.position)]
        _line(right, curve, COLORS[group], group)
        rho, p_value = spearmanr(rows.position, rows.residual)
        stats.append({"panel": "residual", "group": group, "n": len(rows),
                      "median_surprisal": rows[sur].median(), "rho": rho, "p": p_value})

    left.set_ylabel(r"$\|\Delta z\|$", labelpad=8)
    left.set_title(f"Within {sur.replace('s_', '')} surprisal strata", fontsize=10, pad=6)
    right.set_ylabel(r"$\|\Delta z\|$ residual (surprisal removed)", labelpad=8)
    right.set_title("Surprisal regressed out", fontsize=10, pad=6)
    for axis in (left, right):
        axis.set_xlabel("Phoneme position in word", labelpad=8)
        axis.set_xticks(sorted(positions))
        axis.set_xlim(min(positions) - 0.3, max(positions) + 0.3)
        axis.legend(frameon=False, fontsize=8, handlelength=1.4)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        axis.grid(axis="y", alpha=0.25, lw=0.6)
        sns.despine(ax=axis)
    figure.tight_layout()
    return figure, pd.DataFrame(stats)


# --------------------------------------------------------------------------- #
def main(min_count: int = 5, by_type: bool = False):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    tokens = build_table()
    tokens = tokens[tokens.position <= MAX_POS].reset_index(drop=True)
    tokens["type"] = tokens.phoneme_type.map({"C": "Consonant", "V": "Vowel"}).fillna("EOS")
    print(f"{len(tokens):,} delta tokens, positions {tokens.position.min()}-"
          f"{tokens.position.max()}, {tokens.seq_id.nunique():,} words\n")

    regressions, figures = [], {}
    for sur in FIGURE_SURS:
        short = sur.replace("s_", "")
        compare_figure, compare_stats = compare_plot(tokens, sur, min_count)
        table = regression_table(tokens, sur)
        strat_figure, strat_stats = stratified_plot(tokens, sur, min_count=min_count - 1,
                                                    by_type=by_type)
        regressions.append(table)
        figures |= {f"norm_vs_{short}": compare_figure,
                    f"regression_{short}": regression_plot(table),
                    f"strata_{short}": strat_figure}
        print(f"########## {sur} ##########")
        for frame in (compare_stats, table.drop(columns=["surprisal", "n"]), strat_stats):
            print(frame.round(4).to_string(index=False), end="\n\n")

    regressions = pd.concat(regressions, ignore_index=True)
    regressions.to_csv(PLOTS_DIR / "regression.csv", index=False)
    for name, figure in figures.items():
        figure.savefig(PLOTS_DIR / f"{name}.pdf", bbox_inches="tight")
        figure.savefig(PLOTS_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
        plt.close(figure)
    print(f"{len(figures)} figures + regression.csv -> {PLOTS_DIR}")


if __name__ == "__main__":
    main()
