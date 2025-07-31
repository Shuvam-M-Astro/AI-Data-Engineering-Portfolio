import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import warnings
warnings.filterwarnings('ignore')

def convert_to_csv(data_dir, output_csv):
    """Convert IMDB reviews to CSV format"""
    reviews = []
    sentiments = []
    
    try:
        for sentiment in ['pos', 'neg']:
            directory = os.path.join(data_dir, sentiment)
            if not os.path.exists(directory):
                print(f"Directory not found: {directory}")
                return False
                
            for filename in os.listdir(directory):
                if filename.endswith('.txt'):
                    with open(os.path.join(directory, filename), 'r', encoding='utf8') as file:
                        reviews.append(file.read())
                        sentiments.append(1 if sentiment == 'pos' else 0)

        df = pd.DataFrame({
            'review': reviews,
            'sentiment': sentiments
        })

        df.to_csv(output_csv, index=False)
        print(f"Successfully converted {len(df)} reviews to {output_csv}")
        return True
        
    except Exception as e:
        print(f"Error converting data: {str(e)}")
        return False

def create_sample_imdb_data(output_csv, n_samples=1000):
    """Create synthetic IMDB-like data for testing"""
    print("Creating synthetic IMDB-like data...")
    
    np.random.seed(42)
    
    # Sample positive and negative reviews
    positive_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot was engaging from start to finish.",
        "I loved every minute of this film. The cinematography was beautiful and the story was compelling.",
        "A masterpiece of cinema. The director did an incredible job bringing this story to life.",
        "Outstanding performance by all actors. This is definitely one of the best movies I've seen this year.",
        "The screenplay was brilliant and the dialogue was sharp. Highly recommended!",
        "Amazing special effects and great storytelling. This movie exceeded all my expectations.",
        "The character development was excellent and the pacing was perfect. A truly enjoyable experience.",
        "This film has everything: great acting, beautiful visuals, and an unforgettable story.",
        "I was completely immersed in this movie. The atmosphere and mood were perfectly captured.",
        "A cinematic triumph that will be remembered for years to come."
    ]
    
    negative_reviews = [
        "This was one of the worst movies I've ever seen. Terrible acting and a confusing plot.",
        "I can't believe I wasted my time watching this. The story made no sense at all.",
        "Poor direction and even worse acting. This film is a complete disaster.",
        "The dialogue was cringe-worthy and the plot was predictable. Very disappointing.",
        "I expected so much more from this movie. The execution was terrible.",
        "Boring, slow-paced, and completely forgettable. Don't waste your time.",
        "The acting was wooden and the script was awful. This movie is a failure.",
        "I couldn't wait for this movie to end. It was painful to watch.",
        "The special effects were cheap and the story was poorly written. Avoid this film.",
        "A complete waste of time and money. This movie should never have been made."
    ]
    
    reviews = []
    sentiments = []
    
    # Generate positive reviews
    for _ in range(n_samples // 2):
        review = np.random.choice(positive_reviews)
        # Add some variation
        if np.random.random() > 0.7:
            review += " The soundtrack was also amazing."
        if np.random.random() > 0.8:
            review += " I would definitely watch it again."
        reviews.append(review)
        sentiments.append(1)
    
    # Generate negative reviews
    for _ in range(n_samples // 2):
        review = np.random.choice(negative_reviews)
        # Add some variation
        if np.random.random() > 0.7:
            review += " The editing was also terrible."
        if np.random.random() > 0.8:
            review += " I regret watching this movie."
        reviews.append(review)
        sentiments.append(0)
    
    # Shuffle the data
    indices = np.random.permutation(len(reviews))
    reviews = [reviews[i] for i in indices]
    sentiments = [sentiments[i] for i in indices]
    
    df = pd.DataFrame({
        'review': reviews,
        'sentiment': sentiments
    })
    
    df.to_csv(output_csv, index=False)
    print(f"Created synthetic dataset with {len(df)} reviews: {output_csv}")
    print(f"Positive reviews: {sum(sentiments)}")
    print(f"Negative reviews: {len(sentiments) - sum(sentiments)}")
    return True

def main():
    """Main function to handle IMDB data conversion"""
    print("IMDB Dataset Converter")
    print("=" * 50)
    
    # Try different possible paths for the IMDB dataset
    possible_paths = [
        './aclImdb/train',
        './data/aclImdb/train',
        './imdb/train',
        './dataset/aclImdb/train',
        './downloads/aclImdb/train'
    ]
    
    train_success = False
    test_success = False
    
    # Try to find and convert training data
    for path in possible_paths:
        print(f"Trying path: {path}")
        if os.path.exists(path):
            print(f"Found training data at: {path}")
            train_success = convert_to_csv(path, 'imdb_reviews_train.csv')
            if train_success:
                break
    
    # Try to find and convert test data
    for path in possible_paths:
        test_path = path.replace('/train', '/test')
        print(f"Trying test path: {test_path}")
        if os.path.exists(test_path):
            print(f"Found test data at: {test_path}")
            test_success = convert_to_csv(test_path, 'imdb_reviews_test.csv')
            if test_success:
                break
    
    # If no real data found, create synthetic data
    if not train_success:
        print("\nNo IMDB dataset found. Creating synthetic data for testing...")
        create_sample_imdb_data('imdb_reviews_train.csv', n_samples=1000)
    
    if not test_success:
        print("Creating synthetic test data...")
        create_sample_imdb_data('imdb_reviews_test.csv', n_samples=200)
    
    print("\n" + "=" * 50)
    print("Conversion completed!")
    print("Files created:")
    if os.path.exists('imdb_reviews_train.csv'):
        df_train = pd.read_csv('imdb_reviews_train.csv')
        print(f"- imdb_reviews_train.csv ({len(df_train)} reviews)")
    if os.path.exists('imdb_reviews_test.csv'):
        df_test = pd.read_csv('imdb_reviews_test.csv')
        print(f"- imdb_reviews_test.csv ({len(df_test)} reviews)")

if __name__ == "__main__":
    main()
