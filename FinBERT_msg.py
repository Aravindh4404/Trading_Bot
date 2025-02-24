import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from typing import Dict, List, Tuple
import numpy as np


class FinBERTSentimentAnalyzer:
    def __init__(self):
        # Load FinBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        # Financial metrics patterns
        self.financial_metrics = {
            'revenue': r'revenue.*?\$?\s*(\d+\.?\d*)\s*(billion|million|trillion)',
            'profit': r'(profit|loss).*?\$?\s*(\d+\.?\d*)\s*(billion|million|trillion)',
            'growth': r'(increase|decrease|grew|declined).*?(\d+(?:\.\d+)?)%'
        }

    def extract_content_from_json(self, json_data: str) -> List[str]:
        """Extract and clean content from JSON data."""
        try:
            data = json.loads(json_data) if isinstance(json_data, str) else json_data
            messages = [entry["content"] for entry in data["results"]]
            return messages
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return []

    def clean_text(self, text: str) -> str:
        """Clean and prepare text for analysis."""
        # Remove special characters and extra whitespace
        text = re.sub(r'[\[\]\(\)\{\}]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_finbert_sentiment(self, text: str) -> Dict[str, float]:
        """Get FinBERT sentiment scores for a piece of text."""
        # Prepare the text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Convert to numpy for easier handling
        scores = predictions[0].numpy()

        # FinBERT classes are: positive, negative, neutral
        return {
            'positive': float(scores[0]),
            'negative': float(scores[1]),
            'neutral': float(scores[2]),
            'compound': float(scores[0] - scores[1])  # Custom compound score
        }

    def extract_financial_metrics(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract key financial metrics and their context."""
        metrics = {
            'revenue': [],
            'profit': [],
            'growth': [],
            'market_position': []
        }

        # Extract revenue
        revenue_matches = re.finditer(self.financial_metrics['revenue'], text, re.IGNORECASE)
        for match in revenue_matches:
            amount = float(match.group(1))
            unit = match.group(2).lower()
            # Convert to billions for consistency
            if unit == 'million':
                amount /= 1000
            elif unit == 'trillion':
                amount *= 1000
            metrics['revenue'].append(('billion', amount))

        # Extract profit/loss
        profit_matches = re.finditer(self.financial_metrics['profit'], text, re.IGNORECASE)
        for match in profit_matches:
            type_ = match.group(1).lower()
            amount = float(match.group(2))
            unit = match.group(3).lower()
            # Convert to billions for consistency
            if unit == 'million':
                amount /= 1000
            elif unit == 'trillion':
                amount *= 1000
            metrics['profit'].append((type_, amount))

        # Extract growth
        growth_matches = re.finditer(self.financial_metrics['growth'], text, re.IGNORECASE)
        for match in growth_matches:
            direction = match.group(1).lower()
            percentage = float(match.group(2))
            metrics['growth'].append((direction, percentage))

        return metrics

    def analyze_market_sentiment(self, text: str) -> Dict[str, any]:
        """Perform detailed market sentiment analysis using FinBERT."""
        # Clean the text
        cleaned_text = self.clean_text(text)

        # Get FinBERT sentiment scores
        sentiment_scores = self.get_finbert_sentiment(cleaned_text)

        # Extract financial metrics
        financial_metrics = self.extract_financial_metrics(cleaned_text)

        # Generate market signals
        market_signals = self.generate_market_signals(sentiment_scores, financial_metrics)

        return {
            'sentiment_scores': sentiment_scores,
            'financial_metrics': financial_metrics,
            'market_signals': market_signals
        }

    def generate_market_signals(
            self,
            sentiment: Dict[str, float],
            metrics: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, str]:
        """Generate detailed market signals based on FinBERT sentiment and metrics."""
        compound_score = sentiment['compound']

        # Signal strength based on sentiment confidence
        max_sentiment = max(sentiment['positive'], sentiment['negative'], sentiment['neutral'])
        signal_strength = 'strong' if max_sentiment > 0.67 else 'moderate' if max_sentiment > 0.33 else 'weak'

        # Determine action based on sentiment and confidence
        if sentiment['positive'] > max(sentiment['negative'], sentiment['neutral']):
            action = 'BUY'
        elif sentiment['negative'] > max(sentiment['positive'], sentiment['neutral']):
            action = 'SELL'
        else:
            action = 'HOLD'

        # Adjust confidence based on available metrics
        confidence = 'high' if len(metrics['revenue']) > 0 and len(metrics['profit']) > 0 else 'medium'

        return {
            'action': action,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'sentiment_confidence': max_sentiment
        }


def main():
    # Initialize analyzer
    analyzer = FinBERTSentimentAnalyzer()

    try:
        # Read and parse JSON file
        with open('message.txt', 'r', encoding='utf-8') as file:
            json_data = file.read()

        # Extract content
        messages = analyzer.extract_content_from_json(json_data)
        combined_text = ' '.join(messages)

        # Analyze sentiment
        analysis_result = analyzer.analyze_market_sentiment(combined_text)

        # Print detailed results
        print("\n=== FINBERT FINANCIAL SENTIMENT ANALYSIS REPORT ===")
        print(f"\nSentiment Scores:")
        for metric, score in analysis_result['sentiment_scores'].items():
            print(f"{metric}: {score:.3f}")

        print(f"\nMarket Signals:")
        signals = analysis_result['market_signals']
        print(f"Recommended Action: {signals['action']}")
        print(f"Signal Strength: {signals['signal_strength']}")
        print(f"Confidence Level: {signals['confidence']}")
        print(f"Sentiment Confidence: {signals['sentiment_confidence']:.3f}")

        print("\nKey Financial Metrics Found:")
        for metric_type, values in analysis_result['financial_metrics'].items():
            if values:
                print(f"{metric_type}: {values}")

    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()