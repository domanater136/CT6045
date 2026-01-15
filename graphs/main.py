import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))


def load_clean_data(path):
    dataset = pd.read_csv(path)
    dataset['clean_review'] = dataset['clean_review'].astype(str)
    return dataset


def tokenize(text):
    tokens = word_tokenize(text)
    clean_tokens = []
    for token in tokens:
        if token.isalpha() and token not in STOP_WORDS:
            clean_tokens.append(token)
    tokens_set = set(clean_tokens)
    return tokens_set


def build_cooccurrence(dataset, polarity, min_word_freq=40):
    word_counts = defaultdict(int)
    pair_counts = defaultdict(int)

    if polarity == 2:
        reviews = dataset[dataset['polarity'] == 2]
    else:
        reviews = dataset[dataset['polarity'] == 1]

    for index, row in reviews.iterrows():
        review_text = row['clean_review']
        words = tokenize(review_text)

        for word in words:
            word_counts[word] += 1

        words_list = list(words)
        for i in range(len(words_list)):
            for j in range(i + 1, len(words_list)):
                word1 = words_list[i]
                word2 = words_list[j]
                if word1 < word2:
                    pair_key = (word1, word2)
                else:
                    pair_key = (word2, word1)
                pair_counts[pair_key] += 1

    common_words = set()
    for word, count in word_counts.items():
        if count >= min_word_freq:
            common_words.add(word)

    filtered_pairs = {}
    for (word1, word2), pair_count in pair_counts.items():
        if word1 in common_words and word2 in common_words:
            filtered_pairs[(word1, word2)] = pair_count

    return filtered_pairs, word_counts


def build_graph(edges):
    graph = nx.Graph()

    for word_pair, weight in edges.items():
        word1, word2 = word_pair
        graph.add_edge(word1, word2, weight=weight)

    return graph


def plot_graph(graph, word_counts, title, top_n=10):
    node_importance = {}
    for node in graph.nodes():
        if node in word_counts:
            node_importance[node] = word_counts[node]
        else:
            node_importance[node] = 0

    sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)

    top_node_names = []
    for i in range(min(top_n, len(sorted_nodes))):
        node_name, node_importance_value = sorted_nodes[i]
        top_node_names.append(node_name)

    small_graph = graph.subgraph(top_node_names)

    plt.figure(figsize=(12, 10))
    positions = nx.spring_layout(small_graph, k=0.4, seed=42)

    node_importance_values = []
    for node in small_graph.nodes():
        node_importance_values.append(node_importance[node])

    edge_strength_values = []
    for node1, node2 in small_graph.edges():
        edge_strength_values.append(small_graph[node1][node2]['weight'])

    nodes = nx.draw_networkx_nodes(
        small_graph,
        positions,
        node_size=400,
        node_color=node_importance_values,
        cmap=plt.cm.rainbow,
        alpha=0.9
    )

    edges = nx.draw_networkx_edges(
        small_graph,
        positions,
        edge_color=edge_strength_values,
        edge_cmap=plt.cm.rainbow,
        width=1.5,
        alpha=0.6
    )

    nx.draw_networkx_labels(
        small_graph,
        positions,
        font_size=9,
        font_color='black',
        bbox=dict(
            boxstyle='round,pad=0.1',
            facecolor='white',
            edgecolor='none',
        )
    )

    plt.colorbar(nodes, label="Word Frequency")
    plt.colorbar(edges, label="Co-occurrence Count")

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_word_specific_graph(dataset, target_word, polarity=2):
    if polarity == 2:
        polarity_label = "positive"
    else:
        polarity_label = "negative"

    print(f"Building graph for '{target_word}' in {polarity_label} reviews")

    reviews = dataset[dataset['polarity'] == polarity]

    word_counts = defaultdict(int)
    pair_counts = defaultdict(int)

    for index, row in reviews.iterrows():
        review_text = row['clean_review']
        words = tokenize(review_text)

        if target_word in words:
            for word in words:
                word_counts[word] += 1

            words_list = list(words)
            for i in range(len(words_list)):
                for j in range(i + 1, len(words_list)):
                    word1 = words_list[i]
                    word2 = words_list[j]

                    if target_word in (word1, word2):
                        if word1 < word2:
                            pair_key = (word1, word2)
                        else:
                            pair_key = (word2, word1)
                        pair_counts[pair_key] += 1

    target_pairs = {}
    for (word1, word2), count in pair_counts.items():
        if target_word in (word1, word2):
            target_pairs[(word1, word2)] = count

    target_graph = build_graph(target_pairs)
    plot_graph(target_graph, word_counts, f"Word Network for '{target_word}' in {polarity_label.title()} Reviews", top_n=100)


def main():
    print("Loading Data")
    dataset = load_clean_data("amazon_clean_nlp.csv")

    plot_word_specific_graph(dataset, target_word="time", polarity=2)#

    plot_word_specific_graph(dataset, target_word="good", polarity=1)

    print("Building positive review network")
    positive_pairs, positive_word_counts = build_cooccurrence(dataset, polarity=2)
    positive_graph = build_graph(positive_pairs)
    plot_graph(positive_graph, positive_word_counts, "Positive Review Word Network")

    print("Building negative review network")
    negative_pairs, negative_word_counts = build_cooccurrence(dataset, polarity=1)
    negative_graph = build_graph(negative_pairs)
    plot_graph(negative_graph, negative_word_counts, "Negative Review Word Network")


if __name__ == "__main__":
    main()