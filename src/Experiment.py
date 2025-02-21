import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

class Experiment:
    def __init__(self):
        self.conditions = []

    def add_condition(self, sdt_obj, label=None):
        self.conditions.append((sdt_obj, label))

    def sorted_roc_points(self):
        if not self.conditions:
            raise ValueError("No conditions available to generate ROC points.")

        
        roc_points = [(sdt.false_alarm_rate(), sdt.hit_rate()) for sdt, _ in self.conditions]
        roc_points.sort() 

        false_alarm_rates, hit_rates = zip(*roc_points)
        return list(false_alarm_rates), list(hit_rates)

    def compute_auc(self):
        if not self.conditions:
            raise ValueError("No conditions available to compute AUC.")

        false_alarm_rates, hit_rates = self.sorted_roc_points()
        return trapz(hit_rates, false_alarm_rates)  

    def plot_roc_curve(self, show_plot=True):
        false_alarm_rates, hit_rates = self.sorted_roc_points()
        plt.figure()
        plt.plot(false_alarm_rates, hit_rates, marker='o', linestyle='-', label="ROC Curve")
        plt.plot([0, 1], [0, 1], 'k--', label="Chance (AUC=0.5)")
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit Rate")
        plt.title("ROC Curve")
        plt.legend()
        if show_plot:
            plt.show()
