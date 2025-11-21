

class ResultsTally:
    def __init__(self, database):
        self.db = database

    def display_results(self):
        primary_precision = (self.db.primary_hits_count / self.db.total_primary_queries * 100) if self.db.total_primary_queries > 0 else 0
        secondary_precision = (self.db.secondary_hits_count / self.db.total_secondary_queries * 100) if self.db.total_secondary_queries > 0 else 0

        print("----- Precision Testing Results -----")
        print(f"Primary Queries: {self.db.total_primary_queries}, Primary Hits: {self.db.primary_hits_count}, Primary Precision: {primary_precision:.2f}%")
        print(f"Secondary Queries: {self.db.total_secondary_queries}, Secondary Hits: {self.db.secondary_hits_count}, Secondary Precision: {secondary_precision:.2f}%")
        print("-------------------------------------")

    def complete_results(self):
        tp = self.db.true_positives
        tn = self.db.true_negatives
        fp = self.db.false_positives
        fn = self.db.false_negatives
        total_queries = tp + tn + fp + fn

        accuracy = (tp + fn) / (total_queries) # Accuracy is the proportion of all classifications that were correct, whether positive or negative
        tpr = tp / (tp + fn) # true positive rate (TPR), or the proportion of all actual positives that were classified correctly as positives, is also known as recall.
        fpr = fp / (fp + tn) # false positive rate (FPR) is the proportion of all actual negatives that were classified incorrectly as positives, also known as the probability of false alarm
        precision = tp / (tp + fp) # Precision is the proportion of all the model's positive classifications that are actually positive.
        print("\n===== Final Testing Summary =====")

        print(f"TP:{tp} ({tp/total_queries})|TN:{tn} ({tn/total_queries})|FP:{fp} ({fp/total_queries})|FN:{fn} ({fn/total_queries})")
        print("-------------------------------------------")
        print(f"Accuracy: {accuracy*100}%. (All classifications that were correct, whether positive or negative)")
        print(f"Recall/TPR: {tpr*100}%. (tp / tp + fn) Use when false negatives are more expensive than false positives.)")
        print(f"FPR: {fpr*100}%. (fp / fp + tn) Probability of false alarm. Use when false positives are more expensive.)")
        print(f"Precision: {precision*100}%. (tp / tp + fp) Use when it's very important for positive predictions to be accurate.)")
        print("===========================================")