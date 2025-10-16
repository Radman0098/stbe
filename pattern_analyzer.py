import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

def cached_mlpg_score(pattern: tuple) -> float:
    """Placeholder ML-PG score function."""
    return np.random.rand()

class PatternAnalyzer:
    def __init__(self, config: Dict = None):
        default_config = {
            "time_weight_alpha": 0.5,
            "sequence_lengths": [2, 3],
            "analysis_depth": 10,
            "min_pattern_observations": 3,
            "max_history_size": 1000,
            "reliability_adjustment_factor": 0.7,
            "sequence_confidence_weight": 0.6,
            "frequency_confidence_weight": 0.4
        }
        if config:
            default_config.update(config)
        self.config = default_config
        self._last_prob_debug = {}
        self.transition_matrices = {}
        self.global_pattern_stats = {}
        self._internal_log = []

    def _compute_probability_distribution(self, patterns: List[tuple]) -> Dict[tuple, float]:
        unique = sorted(set(patterns), key=str)
        if not unique:
            self._last_prob_debug = {"total_patterns": 0, "sample": []}
            return {}

        # Convert categorical patterns to numerical (+1/-1/0) safely
        numerical_patterns = []
        for p in patterns:
            num_p = []
            for move in p:
                if move == 'U': num_p.append(1)
                elif move == 'D': num_p.append(-1)
                else: num_p.append(0)
            numerical_patterns.append(num_p)

        volatility = np.std([np.mean(p) for p in numerical_patterns]) if numerical_patterns else 1.0
        alpha = self.config["time_weight_alpha"] / (1 + volatility)

        raw = {p: cached_mlpg_score(p) for p in unique}
        time_scores = alpha * np.arange(len(patterns), dtype=np.float64)
        if time_scores.size > 0:
            time_scores -= np.max(time_scores)
        weights = np.exp(time_scores)
        wc = defaultdict(float)
        for i, pat in enumerate(patterns):
            wc[pat] += float(weights[i])
        total = sum(wc.values()) or 1.0
        adjusted = {p: raw[p] * (wc[p]/total) for p in unique}
        vals = np.array(list(adjusted.values()), dtype=np.float64)
        if vals.size == 0:
            self._last_prob_debug = {"total_patterns": len(unique), "sample": unique[:3], "error": "Empty values for softmax"}
            return {}
        e = np.exp(vals - np.max(vals))
        final = e / e.sum()
        prob_map = dict(zip(unique, final))
        self._last_prob_debug = {
            "total_patterns": len(unique),
            "unique_sample": unique[:3],
            "raw_values_sample": list(adjusted.items())[:3]
        }
        return {p: float(prob_map.get(p,0.0)) for p in unique}

    def _analyze_sequences(self) -> Dict[str, Dict[tuple, Dict]]:
        seq_report = {}
        for L in self.config["sequence_lengths"]:
            probs = {}
            if L not in self.transition_matrices:
                continue
            for cur, nxts in self.transition_matrices[L].items():
                total = sum(nxts.values())
                if total <= 0:
                    continue
                probs[cur] = {}
                for p, count in nxts.most_common(3):
                    prob = float(count/total)
                    reliability = "high" if count >= 3 else "low"
                    probs[cur][p] = {
                        "probability": round(prob,4),
                        "reliability": reliability,
                        "count": int(count)
                    }
            seq_report[f"L{L}"] = probs
        return seq_report

    def _sequence_based_prediction(self, last_pattern: tuple, sequence_analysis: Dict, prob_dict: Dict=None) -> Dict:
        best_pred = None
        max_conf = -1.0
        adjust_factor = self.config.get("reliability_adjustment_factor",0.7)

        # exact match
        for seq_key, seq_data in sequence_analysis.items():
            if last_pattern in seq_data:
                predictions = seq_data[last_pattern]
                for next_pattern, meta in predictions.items():
                    confidence = meta.get("probability",0.0)
                    reliability = meta.get("reliability","low")
                    adjusted_conf = confidence * (adjust_factor if reliability=="low" else 1.0)
                    if adjusted_conf > max_conf:
                        max_conf = adjusted_conf
                        best_pred = {
                            "method": f"sequence_{seq_key}",
                            "pattern": str(list(next_pattern)),
                            "confidence": round(adjusted_conf,4),
                            "reliability": reliability,
                            "source": "historical_sequences"
                        }

        # partial match (weighted by overlap)
        if not best_pred:
            for seq_key, seq_data in sequence_analysis.items():
                for seq_pat, nxts in seq_data.items():
                    overlap = sum(1 for i in range(min(len(last_pattern), len(seq_pat))) if last_pattern[i]==seq_pat[i])
                    if overlap == 0:
                        continue
                    for next_pattern, meta in nxts.items():
                        confidence = meta.get("probability",0.0) * (0.5 + 0.5*overlap/len(last_pattern))
                        if confidence > max_conf:
                            max_conf = confidence
                            best_pred = {
                                "method": f"sequence_{seq_key}_partial",
                                "pattern": str(list(next_pattern)),
                                "confidence": round(confidence,4),
                                "reliability": meta.get("reliability","low"),
                                "source": "historical_sequences_partial"
                            }

        # fallback using global probabilities
        if not best_pred and prob_dict:
            best_overall = max(prob_dict.items(), key=lambda x:x[1], default=(None,0))
            if best_overall[0]:
                best_pred = {
                    "method": "probability_fallback",
                    "pattern": str(list(best_overall[0])),
                    "confidence": round(best_overall[1],4),
                    "reliability": "unknown",
                    "source": "global_probabilities"
                }

        self._internal_log.append({"last_pattern": last_pattern, "prediction": best_pred})
        return best_pred

    def _analyze_patterns(self, patterns, original_deltas, prob_dict):
        pattern_freq = Counter(patterns)
        top_by_freq = [p for p,_ in pattern_freq.most_common(self.config["analysis_depth"])]
        top_by_prob = sorted(list(set(patterns)), key=lambda p: prob_dict.get(p,0.0), reverse=True)[:self.config["analysis_depth"]]
        combined = []
        seen = set()
        for p in top_by_freq + top_by_prob:
            if p not in seen:
                combined.append(p)
                seen.add(p)
            if len(combined) >= self.config["analysis_depth"]:
                break

        pattern_indices = defaultdict(list)
        for i,p in enumerate(patterns):
            pattern_indices[p].append(i)

        analysis = {}
        for pattern in combined:
            p_str = str(list(pattern))
            stats = self.global_pattern_stats.get(p_str, {"count":0,"delta_ema":0.0,"vol_ema":0.0})
            idxs = pattern_indices.get(pattern,[])
            current_deltas = np.array([float(original_deltas[i]) for i in idxs]) if idxs else np.array([])
            median_cur = float(np.median(current_deltas)) if current_deltas.size else 0.0
            mean_cur = float(np.mean(current_deltas)) if current_deltas.size else 0.0
            dir_bias = float(np.mean(current_deltas>0)) if current_deltas.size else 0.5
            analysis[p_str] = {
                "count_current_window": len(idxs),
                "total_history_count": int(stats.get("count",0)),
                "probability_score": float(round(prob_dict.get(pattern,0.0),6)),
                "statistics":{
                    "median_delta_current": round(median_cur,6),
                    "mean_delta_current": round(mean_cur,6),
                    "mean_delta_EMA": round(stats.get("delta_ema",0.0),6),
                    "volatility_EMA": round(stats.get("vol_ema",0.0),6)
                },
                "direction_bias_current": round(dir_bias,4)
            }
        return analysis

    def _frequency_based_prediction(self, pattern_analysis):
        if not pattern_analysis:
            return None
        reliable = {p:s for p,s in pattern_analysis.items() if s.get("total_history_count",0)>=self.config["min_pattern_observations"]}
        if not reliable:
            return None
        most_frequent = max(reliable.items(), key=lambda x:x[1].get("total_history_count",0))
        max_possible = max(int(self.config.get("max_history_size",1)),1)
        history_count = int(most_frequent[1].get("total_history_count",0))
        confidence = min(1.0, history_count/max_possible)
        return {
            "method":"frequency",
            "pattern":most_frequent[0],
            "confidence": round(confidence,4),
            "source":"pattern_frequency"
        }

    def combined_prediction(self, last_pattern: tuple, sequence_analysis: Dict, pattern_analysis: Dict, prob_dict: Dict=None):
        seq_pred = self._sequence_based_prediction(last_pattern, sequence_analysis, prob_dict)
        freq_pred = self._frequency_based_prediction(pattern_analysis)
        if not seq_pred and not freq_pred:
            return None

        seq_conf = seq_pred["confidence"] if seq_pred else 0.0
        freq_conf = freq_pred["confidence"] if freq_pred else 0.0
        w_seq = self.config.get("sequence_confidence_weight",0.6)
        w_freq = self.config.get("frequency_confidence_weight",0.4)
        total_conf = seq_conf * w_seq + freq_conf * w_freq

        # Choose prediction with higher base confidence
        chosen_pred = seq_pred if seq_conf >= freq_conf else freq_pred
        if chosen_pred:
            final_pred = chosen_pred.copy()
            final_pred["confidence"] = round(total_conf,4)
            final_pred["combined_source"] = "sequence+frequency"
            self._internal_log.append({
                "last_pattern": last_pattern,
                "seq_conf": seq_conf,
                "freq_conf": freq_conf,
                "chosen_pred": final_pred
            })
            return final_pred
        return None