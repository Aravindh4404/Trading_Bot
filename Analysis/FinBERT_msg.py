import json
import torch
import numpy as np
import re
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class FinBERTSentimentAnalyzer:
    def __init__(self):
        # Load FinBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()

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

        # FinBERT classes are: [0] = positive, [1] = negative, [2] = neutral
        return {
            'positive': float(scores[0]),
            'negative': float(scores[1]),
            'neutral': float(scores[2]),
            'compound': float(scores[0] - scores[1])  # A simple "net positivity" measure
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
            type_ = match.group(1).lower()  # "profit" or "loss"
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

    def analyze_market_sentiment(self, text: str, explain: bool = True) -> Dict[str, any]:
        """
        Perform detailed market sentiment analysis using FinBERT.
        If 'explain' is True, generates a short rationale about key words.
        """
        # Clean the text
        cleaned_text = self.clean_text(text)

        # Get FinBERT sentiment scores
        sentiment_scores = self.get_finbert_sentiment(cleaned_text)

        # Extract financial metrics
        financial_metrics = self.extract_financial_metrics(cleaned_text)

        # Generate market signals
        market_signals = self.generate_market_signals(sentiment_scores, financial_metrics)

        explanation = None
        if explain:
            # Provide a short rationale (top tokens that most affect the negative or positive dimension)
            explanation = self.explain_sentiment(cleaned_text)

        return {
            'sentiment_scores': sentiment_scores,
            'financial_metrics': financial_metrics,
            'market_signals': market_signals,
            'explanation': explanation
        }

    def generate_market_signals(
            self,
            sentiment: Dict[str, float],
            metrics: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, str]:
        """Generate detailed market signals based on FinBERT sentiment and metrics."""
        compound_score = sentiment['compound']

        # The highest single sentiment probability among positive, negative, neutral
        max_sentiment_val = max(sentiment['positive'], sentiment['negative'], sentiment['neutral'])
        max_label = max(sentiment, key=sentiment.get)

        # Simplistic approach to gauge 'signal_strength'
        signal_strength = 'strong' if max_sentiment_val > 0.67 else 'moderate' if max_sentiment_val > 0.33 else 'weak'

        # Determine action based on the dominant sentiment
        if sentiment['positive'] > max(sentiment['negative'], sentiment['neutral']):
            action = 'BUY'
        elif sentiment['negative'] > max(sentiment['positive'], sentiment['neutral']):
            action = 'SELL'
        else:
            action = 'HOLD'

        # Confidence if we have at least some financial data
        confidence = 'high' if (metrics['revenue'] or metrics['profit'] or metrics['growth']) else 'medium'

        return {
            'action': action,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'sentiment_confidence': max_sentiment_val,
            'dominant_sentiment': max_label
        }

    def explain_sentiment(self, text: str, top_n: int = 5) -> str:
        """
        A simple 'leave-one-out' approach to identify tokens that
        contribute most to positive or negative sentiment.

        NOTE: This method can be slow if the text is large.
        """
        # Tokenize text
        encoded = self.tokenizer.encode_plus(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            add_special_tokens=True
        )
        input_ids = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask'][0]

        # Get baseline sentiment
        with torch.no_grad():
            base_outputs = self.model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            base_probs = torch.nn.functional.softmax(base_outputs.logits, dim=-1)[0]
        base_positive = base_probs[0].item()
        base_negative = base_probs[1].item()
        base_neutral = base_probs[2].item()

        # We'll only analyze up to top 30 tokens to save time
        max_tokens = min(len(input_ids), 30)

        token_impacts = []
        for i in range(1, max_tokens - 1):  # skip [CLS] (index 0) and [SEP] (last token)
            original_token_id = input_ids[i].item()
            original_token_str = self.tokenizer.convert_ids_to_tokens([original_token_id])[0]

            # Temporarily mask this token by replacing it with [MASK]
            masked_input_ids = input_ids.clone()
            masked_input_ids[i] = self.tokenizer.mask_token_id

            with torch.no_grad():
                outputs = self.model(masked_input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

            # Difference in negative or positive dimension
            diff_pos = base_positive - probs[0].item()
            diff_neg = base_negative - probs[1].item()

            # We'll store how removing this token affects positive or negative
            token_impacts.append((original_token_str, diff_pos, diff_neg))

        # Sort by absolute difference in negative or positive (which token changed the sentiment the most)
        # We'll pick the largest changes in either positive or negative
        token_impacts.sort(key=lambda x: max(abs(x[1]), abs(x[2])), reverse=True)

        explanation_str = "Top tokens affecting sentiment:\n"
        for tok, dpos, dneg in token_impacts[:top_n]:
            # If dpos is positive => removing token *lowered* positivity => token was boosting positivity
            # If dneg is positive => removing token *lowered* negativity => token was boosting negativity
            if abs(dpos) > abs(dneg):
                if dpos > 0:
                    explanation_str += f" • {tok} → likely **increases** positive sentiment\n"
                else:
                    explanation_str += f" • {tok} → likely **decreases** positive sentiment\n"
            else:
                if dneg > 0:
                    explanation_str += f" • {tok} → likely **increases** negative sentiment\n"
                else:
                    explanation_str += f" • {tok} → likely **decreases** negative sentiment\n"

        return explanation_str


def main():
    # Initialize analyzer
    analyzer = FinBERTSentimentAnalyzer()

    try:
        # Read and parse JSON file
        with open('../message.txt', 'r', encoding='utf-8') as file:
            json_data = file.read()

        # Extract content
        messages = analyzer.extract_content_from_json(json_data)
        combined_text = ' '.join(messages)

        # Analyze sentiment with explanation
        analysis_result = analyzer.analyze_market_sentiment(combined_text, explain=True)

        # Print detailed results
        print("\n=== FINBERT FINANCIAL SENTIMENT ANALYSIS REPORT ===")
        print(f"\nSentiment Scores:")
        for metric, score in analysis_result['sentiment_scores'].items():
            print(f"  {metric}: {score:.3f}")

        print(f"\nMarket Signals:")
        signals = analysis_result['market_signals']
        print(f"  Recommended Action: {signals['action']}")
        print(f"  Signal Strength: {signals['signal_strength']}")
        print(f"  Confidence Level: {signals['confidence']}")
        print(f"  Sentiment Confidence: {signals['sentiment_confidence']:.3f}")
        print(f"  Dominant Sentiment: {signals['dominant_sentiment']}")

        print("\nKey Financial Metrics Found:")
        for metric_type, values in analysis_result['financial_metrics'].items():
            if values:
                print(f"  {metric_type}: {values}")

        # Print explanation for major tokens
        if analysis_result['explanation']:
            print("\nExplanation of Key Tokens Influencing Sentiment:")
            print(analysis_result['explanation'])

    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
