import sys
import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.uic import loadUi  # Load UI from Qt Designer files


class Phase2Analyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load UI designed in Qt Designer (ensure your UI file is named 'main_window.ui')
        loadUi("main_window.ui", self)

        # Connect the browse button to file selection
        self.browse_button.clicked.connect(self.select_dataset)

        # Analyze button triggers the analysis
        self.analyze_button.clicked.connect(self.start_analysis)

        # Data-related variables
        self.data = None
        self.target_column = None
        self.sensitive_attributes = None
        self.model = None

    def select_dataset(self):
        # Open a file dialog to select the dataset
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.datasetpath_textEdit.setText(file_path)  # Show path in the UI
                print("Dataset loaded successfully!")
            except Exception as e:
                self.show_error_message(f"Error loading dataset: {e}")

    def start_analysis(self):
        # Validate inputs
        if self.data is None:
            self.show_error_message("Please select a dataset first.")
            return

        self.target_column = self.targetColumnLineEdit.text().strip()
        print(self.target_column)
        print(list(self.data.columns))
        # if self.target_column not in list(self.data.columns):
        #     self.show_error_message("The target column is not valid. Please check your input.")
        #     return

        sensitive_attrs_input = self.sensitiveAttributesLineEdit.text().strip()
        self.sensitive_attributes = [attr.strip() for attr in sensitive_attrs_input.split(",") if attr.strip()]
        if not all(attr in self.data.columns for attr in self.sensitive_attributes):
            self.show_error_message("Some sensitive attributes are not valid. Please check your input.")
            return
        # print(self.sensitive_attributes)
        # Run the analysis
        try:
            self.run_analysis()
        except Exception as e:
            self.show_error_message(f"Error during analysis: {e}")

    def preprocess_data(self):
        # print(self.data)
        self.data = self.clean_dataframe(self.data)
        print(self.data)

        X = pd.get_dummies(self.data.drop(columns=[self.target_column]), drop_first=True)
        y = self.data[self.target_column]
        print(f"X shape: {X.shape}")  # Number of rows and columns in X
        print(f"y shape: {y.shape}")
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        # print(y_train.shape)
        # print(y_test.shape)
        # print(X_test.shape)
        # print(X_train.shape)

        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        # print(y_test.shape)
        # print(X_test.shape)
        accuracy = accuracy_score(y_test, y_pred)
        # print(y_test.shape)
        # print(X_test.shape)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        return X_test, y_test, y_pred

    def analyze_bias(self, y_true, y_pred, sensitive_features):
        metrics = {
            "Selection Rate": selection_rate,
            "True Positive Rate": true_positive_rate,
        }
        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )
        return metric_frame

    def composite_fairness_score(self, metric_frame):
        scores = []
        for metric_name in metric_frame.metric_names:
            disparity = metric_frame.by_group[metric_name].max() - metric_frame.by_group[metric_name].min()
            scores.append(1 - disparity)  # Higher score = more fairness
        return round(sum(scores) / len(scores), 2)

    def plot_bias(self, metric_frame, sensitive_attribute):
        for metric_name in metric_frame.metric_names:
            plt.figure()
            sns.barplot(x=metric_frame.by_group.index, y=metric_frame.by_group[metric_name])
            plt.title(f"{metric_name} by {sensitive_attribute}")
            plt.xlabel(sensitive_attribute)
            plt.ylabel(metric_name)
            plt.show()

    def explain_model(self, X_test):
        explainer = shap.Explainer(self.model, X_test)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test)

    def run_analysis(self):
        X_test, y_test, y_pred = self.train_model()
        print("fish and eggs")
        for attribute in self.sensitive_attributes:
            print("goats and cows")
            print(f"\nAnalyzing bias for {attribute}:")
            print(X_test.shape)
            print(y_test.shape)
            print(y_pred.shape)
            # error is here
            metric_frame = self.analyze_bias(y_test, y_pred, self.data[attribute])

            fairness_score = self.composite_fairness_score(metric_frame)

            print(metric_frame.by_group)
            result = f"Fairness Score for {attribute}: {fairness_score}"
            self.plot_bias(metric_frame, attribute)
            self.plainTextEdit.setPlainText(result)

        # Explainability
        print("\nGenerating explainability visualizations...")
        result = self.explain_model(X_test)
        self.explanation_textEdit.setPlainText(result)

    def show_error_message(self, message):
        QMessageBox.critical(self, "Error", message)

    def clean_dataframe(self, df, odd_values=['?', '', 'NaN', 'na', 'n/a']):
        """
        Cleans a DataFrame by identifying and handling odd values.

        Parameters:
        - df: The DataFrame to clean.
        - odd_values: A list of odd or invalid values to replace (e.g., '?', empty strings).

        Returns:
        - cleaned_df: A cleaned DataFrame with odd values replaced and dropped.
        """
        # Replace odd values with NaN
        df_replaced = df.replace(odd_values, np.nan)
        #
        # # Count odd values in each column
        # odd_value_counts = df_replaced.isna().sum()

        # Drop rows with any odd (NaN) values
        cleaned_df = df_replaced.dropna()

        return cleaned_df


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Phase2Analyzer()
    window.show()
    sys.exit(app.exec_())
