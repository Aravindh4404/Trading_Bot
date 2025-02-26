import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from typing import Dict, List, Tuple


class FinancialSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

        # Financial-specific sentiment dictionary
        # Positive financial terms
        self.financial_positive = {
            'growth': 2.0,
            'profit': 1.8,
            'revenue': 1.5,
            'expansion': 1.7,
            'investment': 1.4,
            'sales': 1.3,
            'increase': 1.5,
            'earnings': 1.6,
            'dividend': 1.5,
            'bullish': 2.0,
            'outperform': 1.8
        }

        # Negative financial terms
        self.financial_negative = {
            'decrease': 1.5,
            'loss': 1.8,
            'debt': 1.5,
            'lawsuit': 1.7,
            'investigation': 1.6,
            'fine': 1.6,
            'penalty': 1.6,
            'bearish': 2.0,
            'underperform': 1.8,
            'downgrade': 1.7,
            'bankruptcy': 2.0
        }

    def extract_content_from_json(self, json_data: str) -> List[str]:
        """Extract and clean content from JSON data."""
        try:
            data = json.loads(json_data) if isinstance(json_data, str) else json_data
            messages = [entry["content"] for entry in data["results"]]
            return messages
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing JSON: {e}")
            # If JSON parsing fails, try to analyze the text directly
            if isinstance(json_data, str):
                return [json_data]
            return []

    def clean_text(self, text: str) -> str:
        """Clean and prepare text for analysis."""
        # Remove special characters and extra whitespace
        text = re.sub(r'[\[\]\(\)\{\}]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_financial_metrics(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract key financial metrics and their context."""
        metrics = {
            'revenue': [],
            'profit': [],
            'growth': [],
            'market_position': []
        }

        # Revenue pattern with improved regex
        revenue_matches = re.finditer(r'revenue.*?\$?\s*(\d+\.?\d*)\s*(million|billion|trillion)', text, re.IGNORECASE)
        for match in revenue_matches:
            amount = float(match.group(1))
            unit = match.group(2).lower()
            metrics['revenue'].append((unit, amount))

        # Profit/Loss pattern with improved regex
        profit_matches = re.finditer(r'(profit|loss).*?\$?\s*(\d+\.?\d*)\s*(million|billion|trillion)', text,
                                     re.IGNORECASE)
        for match in profit_matches:
            sentiment = match.group(1).lower()
            amount = float(match.group(2))
            metrics['profit'].append((sentiment, amount))

        # Growth indicators with fixed regex
        growth_patterns = ['increase', 'decrease', 'grew', 'declined', 'up', 'down']
        for pattern in growth_patterns:
            matches = re.finditer(f'{pattern}.*?(\\d+(?:\\.\\d+)?)%', text, re.IGNORECASE)
            for match in matches:
                context = text[max(0, match.start() - 50):min(len(text), match.end() + 50)]
                metrics['growth'].append((pattern, float(match.group(1)), context))

        # Market position indicators
        market_terms = ['market share', 'market leader', 'market position', 'competitor']
        for term in market_terms:
            if term in text.lower():
                position_match = re.search(f'{term}.*?(\\b\\w+\\b)', text, re.IGNORECASE)
                if position_match:
                    metrics['market_position'].append((term, position_match.group(1)))

        return metrics

    def analyze_market_sentiment(self, text: str) -> Dict[str, any]:
        """Perform detailed market sentiment analysis."""
        # Clean the text
        cleaned_text = self.clean_text(text)

        # Get base sentiment scores
        base_sentiment = self.analyzer.polarity_scores(cleaned_text)

        # Extract financial metrics
        financial_metrics = self.extract_financial_metrics(cleaned_text)

        # Adjust sentiment based on financial context
        adjusted_sentiment = self.adjust_sentiment_for_financial_context(
            base_sentiment,
            cleaned_text,
            financial_metrics
        )

        # Generate market signals
        market_signals = self.generate_market_signals(adjusted_sentiment, financial_metrics)

        return {
            'base_sentiment': base_sentiment,
            'adjusted_sentiment': adjusted_sentiment,
            'financial_metrics': financial_metrics,
            'market_signals': market_signals,
            'key_sentences': self.extract_key_sentences(cleaned_text)
        }

    def extract_key_sentences(self, text: str) -> List[Dict[str, any]]:
        """Extract and analyze key sentences for more detailed insights."""
        # Split text into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        key_sentences = []

        for sentence in sentences:
            if len(sentence) < 10:  # Skip very short sentences
                continue

            # Analyze sentiment for each sentence
            sentiment = self.analyzer.polarity_scores(sentence)

            # Only include significant sentences (very positive or very negative)
            if abs(sentiment['compound']) > 0.3:
                key_sentences.append({
                    'text': sentence,
                    'sentiment': sentiment['compound'],
                    'type': 'positive' if sentiment['compound'] > 0 else 'negative'
                })

        # Sort by absolute sentiment strength
        key_sentences.sort(key=lambda x: abs(x['sentiment']), reverse=True)

        # Return top 5 most significant sentences
        return key_sentences[:5]

    def adjust_sentiment_for_financial_context(
            self,
            base_sentiment: Dict[str, float],
            text: str,
            metrics: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, float]:
        """Adjust sentiment scores based on financial context."""
        adjusted = base_sentiment.copy()
        text_lower = text.lower()

        # Apply positive financial term boosters
        for word, boost in self.financial_positive.items():
            if word in text_lower:
                # Count occurrences to weight impact
                count = text_lower.count(word)
                # Boost positive sentiment
                adjusted['pos'] += 0.05 * count * boost
                # Reduce negative sentiment slightly
                adjusted['neg'] = max(0, adjusted['neg'] - 0.02 * count)

        # Apply negative financial term boosters
        for word, boost in self.financial_negative.items():
            if word in text_lower:
                # Count occurrences to weight impact
                count = text_lower.count(word)
                # Boost negative sentiment
                adjusted['neg'] += 0.05 * count * boost
                # Reduce positive sentiment slightly
                adjusted['pos'] = max(0, adjusted['pos'] - 0.02 * count)

        # Adjust for revenue trends
        if metrics['revenue']:
            # High revenue is generally positive
            total_revenue = sum(amount for _, amount in metrics['revenue'])
            if total_revenue > 10:  # Billions
                adjusted['pos'] *= 1.2
            elif total_revenue > 100:  # Millions
                adjusted['pos'] *= 1.1

        # Adjust for profit/loss
        if metrics['profit']:
            for type_val, amount in metrics['profit']:
                if type_val == 'profit':
                    adjusted['pos'] *= (1 + min(amount * 0.05, 0.5))
                elif type_val == 'loss':
                    adjusted['neg'] *= (1 + min(amount * 0.05, 0.5))

        # Adjust for growth trends
        if metrics['growth']:
            for pattern, percentage, _ in metrics['growth']:
                if pattern in ['increase', 'grew', 'up']:
                    adjusted['pos'] *= (1 + min(percentage * 0.01, 0.5))
                elif pattern in ['decrease', 'declined', 'down']:
                    adjusted['neg'] *= (1 + min(percentage * 0.01, 0.5))

        # Recalculate compound score
        # Simple approximation based on positive and negative scores
        adjusted['compound'] = (adjusted['pos'] - adjusted['neg']) / (
                    adjusted['pos'] + adjusted['neg'] + adjusted['neu'])

        # Normalize compound score to [-1, 1] range
        adjusted['compound'] = max(min(adjusted['compound'], 1.0), -1.0)

        return adjusted

    def generate_market_signals(
            self,
            sentiment: Dict[str, float],
            metrics: Dict[str, List]
    ) -> Dict[str, any]:
        """Generate detailed market signals based on sentiment and metrics."""
        compound_score = sentiment['compound']

        # Initialize signal strength indicators
        signal_strength = 'strong' if abs(compound_score) > 0.5 else 'moderate' if abs(compound_score) > 0.2 else 'weak'

        # Establish confidence level based on metrics presence and consistency
        has_revenue = len(metrics['revenue']) > 0
        has_profit = len(metrics['profit']) > 0
        has_growth = len(metrics['growth']) > 0

        # Calculate metric consistency
        consistency_score = 0
        if has_growth:
            # Check if growth metrics are consistent (all positive or all negative)
            growth_sentiments = [pattern in ['increase', 'grew', 'up'] for pattern, _, _ in metrics['growth']]
            if all(growth_sentiments) or not any(growth_sentiments):
                consistency_score += 1

        if has_profit:
            # Check if profit metrics are consistent
            profit_sentiments = [type_val == 'profit' for type_val, _ in metrics['profit']]
            if all(profit_sentiments) or not any(profit_sentiments):
                consistency_score += 1

        # Determine confidence based on data richness and consistency
        data_richness = sum([has_revenue, has_profit, has_growth])
        if data_richness >= 2 and consistency_score >= 1:
            confidence = 'high'
        elif data_richness >= 1:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Determine action based on sentiment score and confidence
        if compound_score > 0.3 and confidence != 'low':
            action = 'BUY'
            rationale = "Strongly positive sentiment indicates favorable market conditions"
        elif compound_score < -0.3 and confidence != 'low':
            action = 'SELL'
            rationale = "Strongly negative sentiment suggests unfavorable outlook"
        elif 0.1 <= compound_score <= 0.3:
            action = 'HOLD/BUY'
            rationale = "Moderately positive sentiment with some upside potential"
        elif -0.3 <= compound_score <= -0.1:
            action = 'HOLD/SELL'
            rationale = "Moderately negative sentiment with some downside risk"
        else:
            action = 'HOLD'
            rationale = "Neutral sentiment suggests waiting for clearer signals"

        return {
            'action': action,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'compound_score': compound_score,
            'rationale': rationale,
            'supporting_metrics': {
                'has_revenue_data': has_revenue,
                'has_profit_data': has_profit,
                'has_growth_data': has_growth,
                'metric_consistency': consistency_score
            }
        }


def main():
    # Initialize analyzer
    analyzer = FinancialSentimentAnalyzer()

    try:
        # Try to read and parse JSON file
        try:
            with open('../message.txt', 'r', encoding='utf-8') as file:
                json_data = file.read()

            # Extract content
            messages = analyzer.extract_content_from_json(json_data)
            if not messages:
                raise ValueError("Failed to extract messages from JSON")

        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: {e}. Attempting to use a sample financial text...")
            # Use a sample financial text for demonstration
            messages = ["""
            Apple Inc. reported quarterly revenue of $89.5 billion, an increase of 8% compared to last year.
            The company announced a profit of $22.6 billion, up 12% from the previous quarter.
            Analysts had expected revenue growth of only 5%, so Apple exceeded expectations.
            However, iPhone sales decreased by 3% in emerging markets due to increased competition.
            The board announced a 4% increase in quarterly dividends to shareholders.
            """]

        # Combine all messages for analysis
        combined_text = ' '.join(messages)

        # Analyze sentiment
        analysis_result = analyzer.analyze_market_sentiment(combined_text)

        # Print detailed results
        print("\n=== FINANCIAL SENTIMENT ANALYSIS REPORT ===")
        print(f"\nBase Sentiment Scores:")
        for metric, score in analysis_result['base_sentiment'].items():
            print(f"{metric}: {score:.3f}")

        print(f"\nAdjusted Sentiment Scores:")
        for metric, score in analysis_result['adjusted_sentiment'].items():
            print(f"{metric}: {score:.3f}")

        print(f"\nMarket Signals:")
        signals = analysis_result['market_signals']
        print(f"Recommended Action: {signals['action']}")
        print(f"Signal Strength: {signals['signal_strength']}")
        print(f"Confidence Level: {signals['confidence']}")
        print(f"Rationale: {signals['rationale']}")

        print("\nKey Financial Metrics Found:")
        for metric_type, values in analysis_result['financial_metrics'].items():
            if values:
                print(f"{metric_type}:")
                for value in values:
                    print(f"  - {value}")

        print("\nKey Sentences:")
        for idx, sentence in enumerate(analysis_result['key_sentences'], 1):
            sentiment_type = "POSITIVE" if sentence['sentiment'] > 0 else "NEGATIVE"
            print(f"{idx}. [{sentiment_type} {sentence['sentiment']:.2f}] {sentence['text']}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()