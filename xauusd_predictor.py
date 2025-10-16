import pandas as pd
from collections import Counter, defaultdict
from pattern_analyzer import PatternAnalyzer

class XauUsdPredictor:
    """
    A predictor for XAU/USD price movements based on historical pattern analysis.
    """
    def __init__(self, config=None):
        """
        Initializes the predictor with a PatternAnalyzer instance.
        """
        self.analyzer = PatternAnalyzer(config)
        self.patterns = []
        self.deltas = []

    def _define_patterns_and_deltas(self, df: pd.DataFrame, pattern_length: int = 3, future_steps: int = 1):
        """
        Processes historical OHLC data to generate patterns and corresponding deltas.
        A pattern is a sequence of candle movements (Up/Down).
        A delta is the price change after a pattern occurs.
        """
        df['movement'] = 'U'
        df.loc[df['close'] < df['open'], 'movement'] = 'D'

        self.patterns = []
        self.deltas = []

        for i in range(pattern_length, len(df) - future_steps):
            pattern = tuple(df['movement'].iloc[i - pattern_length : i])
            delta = df['close'].iloc[i + future_steps - 1] - df['close'].iloc[i - 1]
            self.patterns.append(pattern)
            self.deltas.append(delta)

    def train(self, data_path: str, pattern_length: int = 3, future_steps: int = 1):
        """
        Trains the model on historical data from a CSV file.
        """
        df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
        self._define_patterns_and_deltas(df, pattern_length, future_steps)

        # Build transition matrices for sequence analysis
        tm = defaultdict(Counter)
        for i in range(len(self.patterns) - 1):
            current_pattern = self.patterns[i]
            next_pattern = self.patterns[i+1]
            tm[current_pattern][next_pattern] += 1
        self.analyzer.transition_matrices = {pattern_length: tm}

        # Build global stats for frequency analysis
        delta_map = defaultdict(list)
        for p, d in zip(self.patterns, self.deltas):
            delta_map[p].append(d)

        global_stats = {}
        for p, deltas in delta_map.items():
            p_str = str(list(p))
            global_stats[p_str] = {
                "count": len(deltas),
                "delta_ema": pd.Series(deltas).ewm(span=20, adjust=False).mean().iloc[-1],
                "vol_ema": pd.Series(deltas).pow(2).ewm(span=20, adjust=False).mean().iloc[-1]
            }
        self.analyzer.global_pattern_stats = global_stats

        # Run the analyses
        self.prob_dict = self.analyzer._compute_probability_distribution(self.patterns)
        self.sequence_analysis = self.analyzer._analyze_sequences()
        self.pattern_analysis = self.analyzer._analyze_patterns(self.patterns, self.deltas, self.prob_dict)

        print("Training complete.")

    def predict(self, recent_data: pd.DataFrame):
        """
        Predicts the next price movement using a combined prediction method.
        `recent_data` should be a DataFrame with `pattern_length` rows.
        """
        pattern_length = self.analyzer.config["sequence_lengths"][0]
        if len(recent_data) < pattern_length:
            raise ValueError(f"Recent data must have at least {pattern_length} rows.")

        recent_data['movement'] = 'U'
        recent_data.loc[recent_data['close'] < recent_data['open'], 'movement'] = 'D'

        last_pattern = tuple(recent_data['movement'].iloc[-pattern_length:])

        # Use the new combined prediction method, now passing prob_dict
        combined_pred = self.analyzer.combined_prediction(
            last_pattern,
            self.sequence_analysis,
            self.pattern_analysis,
            self.prob_dict
        )

        return {
            "last_pattern": str(list(last_pattern)),
            "prediction": combined_pred
        }

    def generate_signal(self, prediction_result: dict, confidence_threshold: float = 0.5, delta_threshold: float = 0.1):
        """
        Generates a trading signal (BUY, SELL, HOLD) based on the combined prediction.
        """
        prediction = prediction_result.get("prediction")
        if not prediction or prediction.get("confidence", 0) < confidence_threshold:
            return "HOLD"

        predicted_pattern_str = prediction.get("pattern")
        if not predicted_pattern_str:
            return "HOLD"

        pattern_stats = self.pattern_analysis.get(predicted_pattern_str)
        if not pattern_stats:
            return "HOLD" # We have no data on this pattern's historical performance

        mean_delta = pattern_stats.get("statistics", {}).get("mean_delta_EMA", 0.0)

        if mean_delta > delta_threshold:
            return "BUY"
        elif mean_delta < -delta_threshold:
            return "SELL"
        else:
            return "HOLD"

if __name__ == '__main__':
    predictor = XauUsdPredictor(config={
        "sequence_lengths": [3],
        "analysis_depth": 15,
        "min_pattern_observations": 5,
        "sequence_confidence_weight": 0.65, # Give more weight to sequence
        "frequency_confidence_weight": 0.35
    })

    predictor.train('XAUUSD_100k_M1_data.csv', pattern_length=3, future_steps=1)

    df_full = pd.read_csv('XAUUSD_100k_M1_data.csv')
    recent_market_data = df_full.iloc[-4:-1].copy()

    prediction_result = predictor.predict(recent_market_data)

    print("\n--- Combined Prediction ---")
    import json
    print(json.dumps(prediction_result, indent=2))

    signal = predictor.generate_signal(prediction_result, confidence_threshold=0.4, delta_threshold=0.05)
    print(f"\n--- Trading Signal ---")
    print(f"Generated Signal: {signal}")