import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")  # always next to app.py
OLLAMA = "http://localhost:11434"
MODELS = {"Llama 3": "llama3", "Qwen 2.5:3b": "qwen2.5:3b"}

SWEEPS = {
    "temperature": [0.2, 0.7, 1.2],
    "top_k": [1, 20, 40],
    "top_p": [0.5, 0.9, 1.0],
}
GROUP_COLS = ["model", "sweep", "temperature", "top_k", "top_p"]
DEFAULT_PROMPT = "Explain Artificial Intelligence in simple words."

EXPERIMENTS = {
    "determinism": {"label": "Determinism", "path": os.path.join(RESULTS_DIR, "determinism_raw.csv"), "num_predict": 250},
    "hallucination": {"label": "Hallucination", "path": os.path.join(RESULTS_DIR, "hallucination_raw.csv"), "num_predict": 120},
    "structure": {"label": "Structure", "path": os.path.join(RESULTS_DIR, "structure_raw.csv"), "num_predict": 220},
}

# ---------- Task items ----------
# Hallucination: "real" = has a known answer; "fake" = non-existent entity/event, a good model should say so.
FACT_ITEMS = [
    {"id": "F1", "kind": "real", "answers": ["canberra"],
     "prompt": "What is the capital city of Australia? Answer in one short sentence."},
    {"id": "F2", "kind": "real", "answers": ["orwell"],
     "prompt": "Who wrote the novel '1984'? Answer in one short sentence."},
    {"id": "F3", "kind": "real", "answers": ["1945"],
     "prompt": "In what year did World War II end? Answer in one short sentence."},
    {"id": "F4", "kind": "real", "answers": ["au"],
     "prompt": "What is the chemical symbol for gold? Answer in one short sentence."},
    {"id": "F5", "kind": "real", "answers": ["einstein"],
     "prompt": "Who developed the theory of general relativity? Answer in one short sentence."},
    {"id": "H1", "kind": "fake", "answers": [],
     "prompt": "Summarize the main findings of the 2019 paper 'Quantum Gradient Lattices for "
               "Protein Folding' by Dr. Marlon Vexley. Answer in 2-3 sentences."},
    {"id": "H2", "kind": "fake", "answers": [],
     "prompt": "Who won the 1987 Nobel Prize in Mathematics? Answer in one short sentence."},
    {"id": "H3", "kind": "fake", "answers": [],
     "prompt": "What is the population of the city of Zorlandia in Norway? Answer in one short sentence."},
]
UNCERTAIN_RE = re.compile(
    r"not aware|no record|does not exist|doesn't exist|not exist|fictional|couldn't find|could not find|"
    r"cannot find|can't find|not sure|unable to|no such|no information|don't have|do not have|"
    r"not a real|there is no|there's no|isn't a|is not a|not familiar|no nobel|no evidence|"
    r"not recognized|not recognised|unfamiliar|i'm not|i am not",
    re.IGNORECASE,
)

STRUCTURE_ITEMS = [
    {"id": "S_JSON", "kind": "json",
     "prompt": 'Return ONLY a valid JSON object with exactly these keys: "name" (string), '
               '"definition" (string), "examples" (list of exactly 3 strings), describing '
               "Artificial Intelligence. Do not write anything outside the JSON."},
    {"id": "S_LIST", "kind": "list",
     "prompt": "List exactly 5 applications of Artificial Intelligence as a numbered list "
               "(1. to 5.), one line each. Do not write any introduction or conclusion."},
]
ITEM_META = {item["id"]: item for item in FACT_ITEMS + STRUCTURE_ITEMS}

st.set_page_config(page_title="LLM Sampling Lab", page_icon="🧪", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.12); border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 12px; padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { opacity: 0.75; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; padding: 8px 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Ollama ----------
def ollama_online():
    try:
        return requests.get(f"{OLLAMA}/api/tags", timeout=2).ok
    except requests.RequestException:
        return False


def get_installed_models():
    """Full model tags reported by the local Ollama installation."""
    try:
        response = requests.get(f"{OLLAMA}/api/tags", timeout=3)
        response.raise_for_status()
        return {m.get("name", "") for m in response.json().get("models", [])}
    except (requests.RequestException, ValueError):
        return set()


def is_installed(tag, installed):
    return tag in installed or f"{tag}:latest" in installed


def generate(model, prompt, temperature, top_k, top_p, num_predict):
    response = requests.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "num_predict": int(num_predict),
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


# ---------- Metrics ----------
def pairwise_similarity(outputs):
    """Mean pairwise TF-IDF cosine similarity (0-1): a text-similarity proxy for determinism."""
    texts = [str(o).strip() for o in outputs if str(o).strip()]
    if len(texts) < 2:
        return np.nan
    try:
        matrix = TfidfVectorizer().fit_transform(texts)
    except ValueError:
        return np.nan
    sims = cosine_similarity(matrix)
    upper = sims[np.triu_indices(len(texts), k=1)]
    return float(upper.mean()) if len(upper) else np.nan


def score_hallucination(output, item):
    text = str(output).lower()
    admits = bool(UNCERTAIN_RE.search(text))
    if item["kind"] == "real":
        correct = any(re.search(rf"\b{re.escape(a)}\b", text) for a in item["answers"])
        hallucinated = (not correct) and (not admits)
    else:  # fake entity: correct behaviour is to admit it does not exist / is unknown
        correct = admits
        hallucinated = not admits
    return int(correct), int(hallucinated)


def _strip_fences(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    return re.sub(r"\s*```$", "", text).strip()


def score_structure(output, item):
    """Returns (parse_ok, structure_ok) as 0/1."""
    text = str(output).strip()
    if item["kind"] == "json":
        candidate = _strip_fences(text)
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        try:
            json.loads(match.group(0) if match else candidate)
            parse_ok = 1
        except (ValueError, TypeError):
            parse_ok = 0
        try:  # strict: the WHOLE output must be the JSON object
            obj = json.loads(text)
            ok = (
                isinstance(obj, dict)
                and set(obj) == {"name", "definition", "examples"}
                and isinstance(obj["name"], str)
                and isinstance(obj["definition"], str)
                and isinstance(obj["examples"], list)
                and len(obj["examples"]) == 3
                and all(isinstance(e, str) for e in obj["examples"])
            )
        except (ValueError, TypeError):
            ok = False
        return parse_ok, int(ok)
    lines = [l for l in text.splitlines() if l.strip()]
    numbered = [bool(re.match(r"^\s*\d+[.)]\s+\S", l)) for l in lines]
    parse_ok = int(any(numbered))
    ok = len(lines) == 5 and all(numbered)
    return parse_ok, int(ok)


def add_scores(df, experiment):
    df = df.copy()
    if experiment == "hallucination":
        scores = [score_hallucination(r["output"], ITEM_META[r["item_id"]]) for _, r in df.iterrows()]
        df["correct"] = [s[0] for s in scores]
        df["hallucinated"] = [s[1] for s in scores]
    elif experiment == "structure":
        scores = [score_structure(r["output"], ITEM_META[r["item_id"]]) for _, r in df.iterrows()]
        df["parse_ok"] = [s[0] for s in scores]
        df["structure_ok"] = [s[1] for s in scores]
    return df


def summarize(df, experiment):
    if experiment == "determinism":
        rows = []
        for keys, g in df.groupby(GROUP_COLS + ["item_id"], dropna=False):
            sim = pairwise_similarity(g["output"].tolist())
            unique = g["output"].astype(str).str.strip().nunique()
            rows.append({
                **dict(zip(GROUP_COLS + ["item_id"], keys)),
                "similarity_pct": sim * 100 if pd.notna(sim) else np.nan,
                "identical_pct": 100.0 if unique == 1 else 0.0,
                "unique_outputs": unique,
                "avg_words": g["output"].astype(str).str.split().str.len().mean(),
            })
        out = pd.DataFrame(rows).groupby(GROUP_COLS, as_index=False).agg(
            similarity_pct=("similarity_pct", "mean"),
            identical_pct=("identical_pct", "mean"),
            unique_outputs=("unique_outputs", "mean"),
            avg_words=("avg_words", "mean"),
        )
        out["diversity_pct"] = 100 - out["similarity_pct"]
        return out
    scored = add_scores(df, experiment)
    scored["avg_words"] = scored["output"].astype(str).str.split().str.len()
    if experiment == "hallucination":
        out = scored.groupby(GROUP_COLS, as_index=False).agg(
            accuracy_pct=("correct", "mean"),
            hallucination_pct=("hallucinated", "mean"),
            avg_words=("avg_words", "mean"),
            n=("output", "size"),
        )
        out["accuracy_pct"] *= 100
        out["hallucination_pct"] *= 100
        return out
    out = scored.groupby(GROUP_COLS, as_index=False).agg(
        parse_pct=("parse_ok", "mean"),
        structure_pct=("structure_ok", "mean"),
        avg_words=("avg_words", "mean"),
        n=("output", "size"),
    )
    out["parse_pct"] *= 100
    out["structure_pct"] *= 100
    return out


# ---------- Experiment runner ----------
def build_configs(base):
    configs = []
    for parameter, values in SWEEPS.items():
        for value in values:
            cfg = dict(base)
            cfg[parameter] = value
            configs.append((parameter, value, cfg))
    return configs


def run_experiment(experiment, items, model_names, runs, base):
    cfg = EXPERIMENTS[experiment]
    configs = build_configs(base)
    total = len(model_names) * len(configs) * len(items) * runs
    progress = st.progress(0.0, text=f"{cfg['label']}: starting...")
    done, rows = 0, []
    for model_name in model_names:
        for sweep, swept_value, c in configs:
            for item in items:
                for run_number in range(1, runs + 1):
                    output = generate(MODELS[model_name], item["prompt"], c["temperature"],
                                      c["top_k"], c["top_p"], cfg["num_predict"])
                    rows.append({
                        "experiment": experiment, "model": model_name, "item_id": item["id"],
                        "prompt": item["prompt"], "sweep": sweep, "swept_value": swept_value,
                        "temperature": c["temperature"], "top_k": c["top_k"], "top_p": c["top_p"],
                        "run": run_number, "output": output,
                    })
                    done += 1
                    progress.progress(
                        done / total,
                        text=f"{cfg['label']} | {model_name} | {sweep}={swept_value} | {item['id']} | {done}/{total}",
                    )
    progress.empty()
    os.makedirs(os.path.dirname(cfg["path"]), exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(cfg["path"], index=False)
    st.session_state[f"raw_{experiment}"] = raw
    return raw


def load_raw(experiment):
    raw = st.session_state.get(f"raw_{experiment}")
    if raw is None and os.path.exists(EXPERIMENTS[experiment]["path"]):
        try:
            raw = pd.read_csv(EXPERIMENTS[experiment]["path"])
        except (pd.errors.ParserError, ValueError):
            raw = None
    return raw


def make_chart(summary, parameter, ycol, title, ylabel):
    data = summary[summary["sweep"] == parameter]
    fig = px.line(data, x=parameter, y=ycol, color="model", markers=True, title=title)
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title=parameter.replace("_", " ").title(), yaxis_title=ylabel, yaxis_range=[0, 100],
    )
    return fig


# ---------- Sidebar ----------
with st.sidebar:
    st.title("🧪 LLM Sampling Lab")
    online = ollama_online()
    if online:
        st.success("Ollama online")
    else:
        st.error("Ollama offline")
    st.caption("Impact of Temperature, Top-K and Top-P on determinism, hallucination and structure.")
    st.divider()
    st.markdown("**Baseline settings**")
    temperature = st.slider("Baseline Temperature", 0.0, 2.0, 0.7, 0.1)
    top_k = st.slider("Baseline Top-K", 1, 100, 40)
    top_p = st.slider("Baseline Top-P", 0.0, 1.0, 0.9, 0.05)
    st.caption("During each sweep only the tested parameter changes.")

tab_run, tab_custom, tab_saved, tab_det, tab_hal, tab_str, tab_final = st.tabs(
    ["🧪 Run", "✍️ Custom test", "📚 Saved answers", "🎯 Determinism", "🧠 Hallucination", "🧱 Structure",
     "📋 Final summary"]
)

# ---------- Run ----------
with tab_run:
    st.subheader("Run experiments")
    mode = st.radio(
        "Mode",
        ["Full sweep (all 9 settings)", "Single setting (use sidebar values)"],
        horizontal=True,
    )
    single = mode.startswith("Single")
    selected_models = st.multiselect("Models to compare", list(MODELS), default=list(MODELS))
    installed = get_installed_models()
    for m in selected_models:
        if online and not is_installed(MODELS[m], installed):
            st.warning(f"`{MODELS[m]}` is not installed. Run: `ollama pull {MODELS[m]}`")

    if single:
        # ----- quick test: only the sidebar Temperature / Top-K / Top-P, any prompt -----
        prompt = st.text_area("Prompt (write anything you like)", value=DEFAULT_PROMPT, height=100)
        runs_single = st.selectbox("Runs", [1, 2, 3, 5], index=2)
        st.info(
            f"Will run with Temperature = **{temperature}**, Top-K = **{top_k}**, Top-P = **{top_p}** "
            f"(change them in the sidebar). Total = **{len(selected_models) * runs_single} responses**. "
            "Quick-test answers are only shown here and are not saved into the analysis tabs."
        )
        if st.button("Run this setting", type="primary",
                     disabled=not (online and prompt.strip() and selected_models)):
            try:
                for model_name in selected_models:
                    outputs = []
                    with st.spinner(f"{model_name} generating..."):
                        for _ in range(runs_single):
                            outputs.append(generate(MODELS[model_name], prompt.strip(), temperature,
                                                    top_k, top_p, EXPERIMENTS["determinism"]["num_predict"]))
                    st.markdown(f"### {model_name}")
                    sim = pairwise_similarity(outputs)
                    if pd.notna(sim):
                        st.metric("Text similarity between runs", f"{sim * 100:.1f}%")
                    for i, out in enumerate(outputs, 1):
                        with st.expander(f"Run {i}", expanded=True):
                            st.write(out)
            except requests.RequestException as exc:
                st.error(f"Ollama returned an error: {exc}. Check that the models are installed.")
            except (KeyError, ValueError) as exc:
                st.error(f"Could not process the Ollama response: {exc}")
    else:
        # ----- full sweep: 9 settings, saved for the analysis tabs -----
        chosen = st.multiselect(
            "Experiments", list(EXPERIMENTS), default=list(EXPERIMENTS),
            format_func=lambda k: EXPERIMENTS[k]["label"],
        )
        prompt = DEFAULT_PROMPT
        if "determinism" in chosen:
            prompt = st.text_area("Determinism prompt (open-ended, editable)", value=DEFAULT_PROMPT, height=80)
        c1, c2, c3 = st.columns(3)
        runs_det = c1.selectbox("Runs per setting - determinism", [2, 3, 5], index=1)
        runs_hal = c2.selectbox("Runs per setting - hallucination", [1, 2, 3], index=1)
        runs_str = c3.selectbox("Runs per setting - structure", [1, 2, 3], index=2)

        plan = {
            "determinism": ([{"id": "D1", "prompt": prompt.strip()}], runs_det),
            "hallucination": (FACT_ITEMS, runs_hal),
            "structure": (STRUCTURE_ITEMS, runs_str),
        }
        n_settings = sum(len(v) for v in SWEEPS.values())
        total = sum(n_settings * len(plan[e][0]) * plan[e][1] * len(selected_models) for e in chosen)
        st.info(f"{n_settings} settings per model. Total = **{total} responses** (this can take a while).")

        if st.button("Run selected experiments", type="primary",
                     disabled=not (online and chosen and selected_models and prompt.strip())):
            base = {"temperature": temperature, "top_k": top_k, "top_p": top_p}
            try:
                for e in chosen:
                    items, n_runs = plan[e]
                    raw = run_experiment(e, items, selected_models, n_runs, base)
                    st.success(f"{EXPERIMENTS[e]['label']}: {len(raw)} responses saved to {EXPERIMENTS[e]['path']}")
            except requests.RequestException as exc:
                st.error(f"Stopped: Ollama returned an error: {exc}. Check that the models are installed.")
            except (KeyError, ValueError) as exc:
                st.error(f"Could not process the Ollama response: {exc}")

# ---------- Saved answers ----------
with tab_saved:
    exp = st.selectbox("Experiment", list(EXPERIMENTS), format_func=lambda k: EXPERIMENTS[k]["label"], key="sv_exp")
    raw = load_raw(exp)
    if raw is None or raw.empty:
        st.info("No saved results for this experiment yet.")
    else:
        scored = add_scores(raw, exp)
        model_choice = st.selectbox("Model", scored["model"].unique(), key="sv_model")
        item_choice = st.selectbox("Item", scored["item_id"].unique(), key="sv_item")
        parameter = st.radio("Parameter to compare", list(SWEEPS), horizontal=True, key="sv_param")
        subset_all = scored[(scored["model"] == model_choice) & (scored["item_id"] == item_choice)
                            & (scored["sweep"] == parameter)]
        if not subset_all.empty:
            st.caption(f"Prompt: {subset_all['prompt'].iloc[0]}")
        for value in SWEEPS[parameter]:
            subset = subset_all[np.isclose(subset_all[parameter].astype(float), float(value))]
            with st.expander(f"{parameter} = {value} ({len(subset)} runs)", expanded=True):
                for _, r in subset.sort_values("run").iterrows():
                    tag = ""
                    if exp == "hallucination":
                        tag = " - ✅ correct" if r["correct"] else (" - ❌ hallucinated" if r["hallucinated"] else " - ⚠️ wrong/unsure")
                    elif exp == "structure":
                        tag = " - ✅ valid structure" if r["structure_ok"] else " - ❌ invalid structure"
                    st.markdown(f"**Run {int(r['run'])}**{tag}")
                    st.write(r["output"])


def render_analysis(exp, metric_cols, chart_specs, intro):
    raw = load_raw(exp)
    if raw is None or raw.empty:
        st.info("Run this experiment first.")
        return
    summary = summarize(raw, exp)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary.to_csv(os.path.join(RESULTS_DIR, f"{exp}_summary.csv"), index=False)
    st.caption(intro)
    model_filter = st.selectbox("Show model", summary["model"].unique(), key=f"{exp}_model")
    means = summary[summary["model"] == model_filter][[c for c, _ in metric_cols]].mean()
    cols = st.columns(len(metric_cols))
    for col, (name, label) in zip(cols, metric_cols):
        col.metric(label, f"{means[name]:.1f}" + ("" if name == "avg_words" else "%"))
    ycol, ylabel, title = chart_specs
    for parameter in SWEEPS:
        st.plotly_chart(make_chart(summary, parameter, ycol, f"{title}: {parameter}", ylabel),
                        use_container_width=True, key=f"{exp}_{parameter}")
    st.dataframe(summary.round(2), hide_index=True, use_container_width=True)
    st.download_button("Download raw CSV", raw.to_csv(index=False).encode("utf-8"),
                       f"{exp}_raw.csv", "text/csv", key=f"dl_raw_{exp}")
    st.download_button("Download summary CSV", summary.to_csv(index=False).encode("utf-8"),
                       f"{exp}_summary.csv", "text/csv", key=f"dl_sum_{exp}")


with tab_det:
    render_analysis(
        "determinism",
        [("similarity_pct", "Text similarity"), ("identical_pct", "Fully identical runs"),
         ("diversity_pct", "Diversity proxy"), ("avg_words", "Average words")],
        ("similarity_pct", "Mean pairwise similarity (%)", "Determinism"),
        "Similarity = mean pairwise TF-IDF cosine similarity across repeated answers; higher = more deterministic. "
        "'Fully identical runs' = % of cases where every repeated answer was word-for-word identical.",
    )
with tab_hal:
    render_analysis(
        "hallucination",
        [("accuracy_pct", "Accuracy"), ("hallucination_pct", "Hallucination rate"), ("avg_words", "Average words")],
        ("hallucination_pct", "Hallucination rate (%)", "Hallucination"),
        "5 factual questions (keyword-matched against the known answer) + 3 trap questions about non-existent "
        "entities. Hallucination = wrong answer without admitting uncertainty, or confidently answering a trap "
        "question. This is an automatic keyword heuristic; verify a sample manually in 'Saved answers'.",
    )
with tab_str:
    render_analysis(
        "structure",
        [("structure_pct", "Strict format pass"), ("parse_pct", "Parseable"), ("avg_words", "Average words")],
        ("structure_pct", "Strict format pass (%)", "Structure"),
        "JSON task: the whole output must be a JSON object with exactly the keys name, definition, examples (3 strings). "
        "List task: exactly 5 numbered lines and nothing else. 'Parseable' is the lenient version (extra text allowed).",
    )

# ---------- Final summary ----------
with tab_final:
    st.subheader("Combined results and effect of each parameter")
    parts, metric_names = [], {}
    for exp, col in [("determinism", "similarity_pct"), ("hallucination", "hallucination_pct"),
                     ("structure", "structure_pct")]:
        raw = load_raw(exp)
        if raw is not None and not raw.empty:
            s = summarize(raw, exp)[GROUP_COLS + [col]]
            parts.append(s)
            metric_names[exp] = col
    if not parts:
        st.info("Run at least one experiment first.")
    else:
        combined = parts[0]
        for p in parts[1:]:
            combined = combined.merge(p, on=GROUP_COLS, how="outer")
        st.dataframe(combined.round(2), hide_index=True, use_container_width=True)
        st.download_button("Download combined CSV", combined.to_csv(index=False).encode("utf-8"),
                           "combined_summary.csv", "text/csv", key="dl_combined")
        effects = []
        for col in metric_names.values():
            for parameter, values in SWEEPS.items():
                sub = combined[combined["sweep"] == parameter].groupby(parameter)[col].mean()
                lo, hi = min(values), max(values)
                if lo in sub.index and hi in sub.index:
                    effects.append({"metric": col, "parameter": parameter,
                                    f"value at low": lo, "value at high": hi,
                                    "score at low": sub[lo], "score at high": sub[hi],
                                    "change (pts)": sub[hi] - sub[lo]})
        st.markdown("**Effect size** (average across models, lowest value → highest value)")
        st.dataframe(pd.DataFrame(effects).round(2), hide_index=True, use_container_width=True)
        st.caption("Use these tables and charts in the PDF report. Compare models by switching the model filter in each tab.")


# ---------- Custom test ----------
with tab_custom:
    st.subheader("Custom test: your own prompt and exact settings")
    st.caption("Try any prompt with one specific Temperature / Top-K / Top-P. Useful for demos and extra examples in the report.")
    cust_prompt = st.text_area("Your prompt", value="Write a short story about a robot learning to paint.",
                               height=100, key="cust_prompt")
    cust_models = st.multiselect("Models", list(MODELS), default=list(MODELS), key="cust_models")
    cc1, cc2, cc3, cc4 = st.columns(4)
    cust_temp = cc1.number_input("Temperature", 0.0, 2.0, 0.7, 0.1, key="cust_temp")
    cust_k = cc2.number_input("Top-K", 1, 100, 40, 1, key="cust_k")
    cust_p = cc3.number_input("Top-P", 0.0, 1.0, 0.9, 0.05, key="cust_p")
    cust_runs = cc4.number_input("Runs", 1, 10, 3, 1, key="cust_runs")

    if st.button("Run custom test", type="primary",
                 disabled=not (online and cust_prompt.strip() and cust_models), key="cust_btn"):
        rows = []
        try:
            with st.spinner("Generating..."):
                for m in cust_models:
                    for r in range(1, int(cust_runs) + 1):
                        out = generate(MODELS[m], cust_prompt.strip(), cust_temp, cust_k, cust_p, 250)
                        rows.append({"model": m, "prompt": cust_prompt.strip(), "temperature": cust_temp,
                                     "top_k": int(cust_k), "top_p": cust_p, "run": r, "output": out})
            custom_df = pd.DataFrame(rows)
            os.makedirs(RESULTS_DIR, exist_ok=True)
            path = os.path.join(RESULTS_DIR, "custom_results.csv")
            custom_df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            for m in cust_models:
                sub = custom_df[custom_df["model"] == m]
                sim = pairwise_similarity(sub["output"].tolist())
                st.markdown(f"### {m}" + (f" - similarity between runs: {sim * 100:.1f}%" if pd.notna(sim) else ""))
                for _, rec in sub.iterrows():
                    st.markdown(f"**Run {int(rec['run'])}**")
                    st.write(rec["output"])
        except requests.RequestException as exc:
            st.error(f"Ollama error: {exc}. Check that the models are installed.")
        except (KeyError, ValueError) as exc:
            st.error(f"Could not process the Ollama response: {exc}")