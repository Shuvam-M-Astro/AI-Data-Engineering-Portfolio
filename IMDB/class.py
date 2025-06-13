import pandas as pd
import numpy as np
import time
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description='IMDB Sentiment Analysis with Advanced Features')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'benchmark'],
                      help='Mode to run the script in')
    parser.add_argument('--quantize', action='store_true', help='Enable quantization')
    parser.add_argument('--prune', action='store_true', help='Enable pruning')
    parser.add_argument('--prune_amount', type=float, default=0.3, help='Amount of pruning (0-1)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--vectorizer', type=str, default='tfidf', choices=['count', 'tfidf'],
                      help='Type of vectorizer to use')
    return parser.parse_args()

class ModelBenchmark:
    def __init__(self, model, device_type):
        self.model = model
        self.device_type = device_type
        self.results = []
        
    def log_metric(self, metric_name, value, metadata=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = {
            'timestamp': timestamp,
            'metric': metric_name,
            'value': value,
            'device': self.device_type,
            'model_size': self.get_model_size(),
            'metadata': metadata or {}
        }
        self.results.append(entry)
        
    def get_model_size(self):
        if hasattr(self.model, 'feature_count_'):
            return self.model.feature_count_.nbytes / 1024  # Size in KB
        return 0
        
    def save_results(self, filename='imdb_benchmark_results.csv'):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def setup_environment():
    return "CPU"  # IMDB uses CPU-based sklearn models

def get_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    X = df['review']
    y = df['sentiment']
    return X, y

def get_vectorizer(vectorizer_type):
    if vectorizer_type == 'count':
        return CountVectorizer(max_features=10000)
    else:
        return TfidfVectorizer(max_features=10000)

def train_model(X, y, args, benchmark):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    # Vectorize the text
    vectorizer = get_vectorizer(args.vectorizer)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Initialize and train the model
    start_time = time.time()
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    benchmark.log_metric('training_time', training_time)
    benchmark.log_metric('accuracy', accuracy)
    
    return model, vectorizer, X_test_vec, y_test

def save_model(model, vectorizer, filename='imdb_model.joblib'):
    joblib.dump({'model': model, 'vectorizer': vectorizer}, filename)
    print(f"Model saved to {filename}")

def load_model(filename='imdb_model.joblib'):
    saved_data = joblib.load(filename)
    return saved_data['model'], saved_data['vectorizer']

def quantize_model(model):
    print("\nQuantizing model probabilities...")
    model.feature_log_prob_ = np.round(model.feature_log_prob_ * 100) / 100
    model.class_log_prior_ = np.round(model.class_log_prior_ * 100) / 100
    return model

def prune_model(model, amount):
    print(f"\nPruning model features by {amount*100}%...")
    threshold = np.percentile(np.abs(model.feature_log_prob_), amount * 100)
    model.feature_log_prob_[np.abs(model.feature_log_prob_) < threshold] = 0
    return model

def benchmark_inference(model, vectorizer, X_test, y_test, benchmark):
    print("\nRunning inference benchmark...")
    
    # Warmup
    for _ in range(10):
        _ = model.predict(vectorizer.transform([X_test[0]]))
    
    # Benchmark
    total_time = 0
    total_samples = len(X_test)
    
    start_time = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, predictions)
    samples_per_second = total_samples / inference_time
    
    print(f"Average inference time per sample: {(inference_time/total_samples)*1000:.2f}ms")
    print(f"Samples per second: {samples_per_second:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    benchmark.log_metric('inference_throughput', samples_per_second)
    benchmark.log_metric('avg_inference_time', inference_time/total_samples)
    benchmark.log_metric('inference_accuracy', accuracy)

def main():
    args = parse_args()
    device_type = setup_environment()
    
    # Create benchmark instance
    benchmark = ModelBenchmark(None, device_type)
    
    if args.mode == 'train':
        X, y = get_data('imdb_reviews_train.csv')
        model, vectorizer, X_test, y_test = train_model(X, y, args, benchmark)
        
        if args.quantize:
            model = quantize_model(model)
            benchmark.model = model
            benchmark.log_metric('model_size_after_quantization', benchmark.get_model_size())
        
        if args.prune:
            model = prune_model(model, args.prune_amount)
            benchmark.model = model
            benchmark.log_metric('model_size_after_pruning', benchmark.get_model_size())
        
        save_model(model, vectorizer)
    
    elif args.mode == 'inference':
        model, vectorizer = load_model()
        benchmark.model = model
        
        if args.quantize:
            model = quantize_model(model)
        if args.prune:
            model = prune_model(model, args.prune_amount)
        
        # Load test data
        X, y = get_data('imdb_reviews_test.csv')
        _, X_test, _, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
        X_test_vec = vectorizer.transform(X_test)
        
        benchmark_inference(model, vectorizer, X_test, y_test, benchmark)
    
    elif args.mode == 'benchmark':
        model, vectorizer = load_model()
        benchmark.model = model
        
        # Load test data
        X, y = get_data('imdb_reviews_test.csv')
        _, X_test, _, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
        X_test_vec = vectorizer.transform(X_test)
        
        # Benchmark original model
        print("\nBenchmarking original model...")
        benchmark_inference(model, vectorizer, X_test, y_test, benchmark)
        
        # Benchmark quantized model
        quantized_model = quantize_model(model)
        print("\nBenchmarking quantized model...")
        benchmark_inference(quantized_model, vectorizer, X_test, y_test, benchmark)
        
        # Benchmark pruned model
        pruned_model = prune_model(model, args.prune_amount)
        print("\nBenchmarking pruned model...")
        benchmark_inference(pruned_model, vectorizer, X_test, y_test, benchmark)
    
    # Save benchmark results
    benchmark.save_results()

if __name__ == '__main__':
    main()
