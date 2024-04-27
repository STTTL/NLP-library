import matplotlib.pyplot as plt
import plotly.graph_objs as go
from collections import defaultdict, Counter
import textstat
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.colors as mcolors
import string

class TextAnalyzer:
    def __init__(self):
        """
        Constructor to initialize the TextAnalyzer class.
        """
        self.data = defaultdict(dict)
        self.stop_words = set()

    def calculate_ttr(self, text):
        """
        Calculates the Type-Token Ratio (TTR) for a given text.
        :param text: Text to analyze
        :return: Type-Token Ratio
        """
        words = text.split()
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0


    @staticmethod
    def _default_parser(filename):
        """
        Default parser for processing simple unformatted text files.
        :param filename: The path to the text file
        :return: Dictionary of parsing results (e.g., word counts)
        """
        with open(filename, 'r', encoding='utf-8') as file:
            # Read text and convert to lowercase
            text = file.read().lower()

            # Remove punctuation
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)

            # Tokenize and count words
            words = text.split()
            word_count = Counter(words)
            return {'wordcount': word_count, 'numwords': len(words)}

    def load_text(self, filename, label=None, parser=None):
        """
        Loads a text file into the library. A custom parser can be specified.
        Text is preprocessed to remove punctuation and convert to lowercase.

        :param filename: The path to the text file
        :param label: An optional label for the text
        :param parser: An optional custom parser function
        """
        if parser is None:
            parser = self._default_parser

        if label is None:
            label = filename

        # Process and store the text
        results = parser(filename)
        for key, value in results.items():
            self.data[key][label] = value


    def load_stop_words(self, stopfile):
        """
        Loads stop words from a file and removes them from the loaded texts,
        regardless of their case.

        :param stopfile: The path to the file containing stop words
        """

        with open(stopfile, 'r') as file:
            self.stop_words = set(word.lower() for word in file.read().splitlines())

        for label, word_count in self.data['wordcount'].items():
            # Convert the keys of the Counter (words) to a set
            word_set = set(word_count.keys())

            # Use set difference to remove stop words
            words_without_stopwords = word_set - self.stop_words

            # Create a new Counter with only the non-stop words
            self.data['wordcount'][label] = Counter({word: word_count[word] for word in words_without_stopwords})

    def wordcount_sankey(self, k=5):
        """
        Creates a word count Sankey diagram for the k most common words across all texts.
        :param k: Number of most common words to include
        """
        # Combine all word counts to find the k most common words
        combined_word_counts = Counter()
        for word_counts in self.data['wordcount'].values():
            combined_word_counts += word_counts
        most_common_words = combined_word_counts.most_common(k)

        # Prepare data for the Sankey diagram
        nodes = []
        source_indices = []
        target_indices = []
        flow_values = []

        # Add nodes for each text and each common word
        text_indices = {label: idx for idx, label in enumerate(self.data['wordcount'].keys())}
        nodes.extend(text_indices.keys())
        word_indices = {word: idx + len(text_indices) for idx, (word, _) in enumerate(most_common_words)}
        nodes.extend(word for word, _ in most_common_words)

        # Add links from texts to words
        for text_label, word_counts in self.data['wordcount'].items():
            for word, count in most_common_words:
                if word in word_counts:
                    source_indices.append(text_indices[text_label])
                    target_indices.append(word_indices[word])
                    flow_values.append(word_counts[word])

        # Create Sankey diagram
        link = {'source': source_indices, 'target': target_indices, 'value': flow_values}
        node = {'label': nodes}
        data = go.Sankey(link=link, node=node)

        # Plot the diagram
        fig = go.Figure(data)
        fig.show()

    def compare_texts_visualization(self):
        """
        Compares the vocabulary richness and readability across loaded texts.
        Plots the Type-Token Ratio and Flesch Reading Ease score for each text.
        """
        labels = []
        ttrs = []
        readability_scores = []

        # Calculate TTR and readability for each text
        for label, content in self.data['wordcount'].items():
            # Join words to form the full text for readability calculation
            full_text = ' '.join(content)
            labels.append(label)
            ttrs.append(self.calculate_ttr(full_text))
            readability_scores.append(textstat.flesch_reading_ease(full_text))

        # Plotting
        x = range(len(labels))  # Label locations
        fig, ax1 = plt.subplots()

        # TTR Plot
        color = 'tab:red'
        ax1.set_xlabel('Texts')
        ax1.set_ylabel('Type-Token Ratio (TTR)', color=color)
        ax1.bar(x, ttrs, color=color, width=0.4, label='TTR', align='center')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')

        # Readability Plot
        ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Flesch Reading Ease', color=color)
        ax2.plot(x, readability_scores, color=color, marker='o', linestyle='-', label='Readability')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # To adjust subplot parameters to give specified padding
        plt.show()

    def calculate_cosine_similarity_matrix(self):
        """
        Calculates cosine similarity between all pairs of loaded texts.
        """
        vectorizer = CountVectorizer()
        # Join the word counts into single strings per text
        texts = [' '.join(content.keys()) for content in self.data['wordcount'].values()]
        text_vectors = vectorizer.fit_transform(texts).toarray()

        # Calculate cosine similarity
        sim_matrix = cosine_similarity(text_vectors)

        return sim_matrix


    def plot_similarity_matrix(self):
        """
        Plots a matrix of cosine similarities between texts with a fixed color scale.
        Labels on the x-axis are rotated and positioned at the bottom.
        The color bar is also named to reflect the scale of similarity.
        """
        sim_matrix = self.calculate_cosine_similarity_matrix()
        labels = list(self.data['wordcount'].keys())
        fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size as needed

        # Define a fixed range for the color scale
        vmin = 0  # Minimum value of data
        vmax = 1  # Maximum value of data

        # Use a colormap and normalize with the fixed range
        cmap = plt.cm.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        cax = ax.matshow(sim_matrix, cmap=cmap, norm=norm, interpolation='nearest')

        # Add a color bar with the fixed range and a label
        color_bar = fig.colorbar(cax, ticks=np.linspace(vmin, vmax, num=5))
        color_bar.set_label('Cosine Similarity')

        # Set tick labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))

        # Rotate x-axis labels and move them to the bottom
        ax.set_xticklabels(labels, rotation=30, rotation_mode='anchor')
        ax.set_yticklabels(labels)

        # Adjust the subplot parameters to move the plot within the figure window
        plt.subplots_adjust(
            top=0.8,  # Reduce the top margin to prevent the plot from going off the page
            bottom=0.4,  # Increase the bottom margin to accommodate x-axis labels
        )

        # Ensure there is enough space for the x-axis labels
        plt.gcf().subplots_adjust(bottom=0.2)

        plt.title('Cosine Similarity between Texts')
        plt.xlabel('Texts')
        plt.ylabel('Texts')
        plt.show()


# Example usage
if __name__ == "__main__":
    analyzer = TextAnalyzer()

    # Load texts
    analyzer.load_text("pg3176.txt", label="The Innocents Abroad")
    analyzer.load_text("pg76.txt", label="Adventures of Huckleberry Finn")
    analyzer.load_text("A Connecticut Yankee in King Arthur’s Court.txt", label="A Connecticut Yankee")
    analyzer.load_text("The £1,000,000 bank-note.txt", label="£1,000,000")
    analyzer.load_text("The-Tragedy-of-Pudd-nhead-Wilson-by-Mark.txt", label="The Tragedy")
    # ... load more texts as needed ...

    # Remove stop words
    analyzer.load_stop_words("NLTK's list of english stopwords.txt")

    # Visualize a Word Count Sankey Diagram for the most common words
    analyzer.wordcount_sankey()

    # Visualize comparison of Type-Token Ratio and Flesch Reading Ease
    # analyzer.compare_texts_visualization()

    # Plot the Cosine Similarity Matrix
    # analyzer.plot_similarity_matrix()

    # print(analyzer.data)
