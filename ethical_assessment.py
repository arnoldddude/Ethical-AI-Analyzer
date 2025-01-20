# phase 2
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import shap
import seaborn as sns
import matplotlib.pyplot as plt

class Phase_2:
    def __init__(self, data, target_column):
        self.dataset_path = data
        self.target_column = target_column
        self.model = None

    def preprocess_data(self):
        X = pd.get_dummies(self.dataset_path.drop(columns=[self.target_column]), drop_first=True)
        y = self.dataset_path[self.target_column]
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_text, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        return X_test, y_test, y_pred

    def analyze_bias(self, y_true, y_pred, sensitive_features):
        metrics = {
            "Selection Rate": selection_rate,
            "True Positive Rate": true_positive_rate,
        }
        # calculate bias metrics
        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        return metric_frame

    def composite_fairness_score(self, metric_frame):
        # Calculate fairness score for each metric
        scores = []
        for metric_name in metric_frame.metric_names:
            disparity = metric_frame.by_group[metric_name].max() - metric_frame.by_group[metric_name].min()
            scores.append(1 - disparity)  # Higher score = more fairness

        # Return average fairness score
        return round(sum(scores) / len(scores), 2)

    # visualize bias using charts
    def plot_bias(self, metric_frame, sensitive_attributes):
        for metric_name in metric_frame.metric_names:
            plt.figure()
            sns.barplot(x=metric_frame.by_group.index,
                        y=metric_frame.by_group[metric_name])
            plt.title(f"{metric_name} by {sensitive_attributes}")
            plt.xlabel(sensitive_attributes)
            plt.ylabel(metric_name)
            plt.show()

    def explain_model(self, X_test):
        # Initialize SHAP explainer
        explainer = shap.Explainer(self.model, X_test)
        shap_values = explainer(X_test)
        # Visualize feature importance
        shap.summary_plot(shap_values, X_test)
        return shap_values

    def run_analysis(self, sensitive_attributes):
    #     Train model and get predictions
        X_test, y_test, y_pred = self.train_model()

        for attribute in sensitive_attributes:
            print(f"\nAnalyzing biad for {attribute}:")
            metric_frame = self.analyze_bias(y_test, y_pred, self.data[attribute])
            fairness = self.fairness_score(metric_frame)

            print(metric_frame.by_group)
            print(f"Fairness Score for {attribute}: {fairness}%")
            self.plot_bias(metric_frame, atribute)

    #   Explainability
        print("\nGenerating explainability visualizations...")
        self.explain_model(X_test)


def main():
    pass

if __name__ == "__main__":
    main()






    print("Choose a sensitive attribute to analyze: [1] Gender, [2] Race, [3] Age")
    choice = int(input("Enter your choice: "))
    sensitive_attribute = sensitive_attributes[choice - 1]

    print("Choose a fairness metric: [1] Selection Rate, [2] True Positive Rate")
    metric_choice = int(input("Enter your choice: "))
    metric = "Selection Rate" if metric_choice == 1 else "True Positive Rate"

    # Run analysis
    metric_frame = analyze_bias(y_true, y_pred, data[sensitive_attribute])
    print(f"{metric} by {sensitive_attribute}:")
    print(metric_frame.by_group)




    # Load dataset
    data = pd.read_csv(dataseth_path)
    print(data.head())

    # WILL USE FOR IMPLEMENTATION OF ANALYZE_BIAS FUNCTION
    # sensitive_attributes = ['gender', 'race', 'age']
    #
    # for attribute in sensitive_attributes:
    #     metric_frame = analyze_bias(y_true, y_pred, data[attribute])
    #     print(f"Metrics by {attribute}:")
    #     print(metric_frame.by_group)

    # defining metrics for bias
    # y_true = data['income']  # Ground truth labels
    # y_pred = model.predict(data.drop(['income', 'gender'], axis=1))  # Predictions
    # sensitive_attribute = data['gender']