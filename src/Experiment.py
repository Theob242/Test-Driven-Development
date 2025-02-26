import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid 
from SignalDetection import SignalDetection as sdt #someone checks if this works

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
        roc_points = [(sdt.false_alarm_rate(), sdt.hit_rate()) for sdt, _ in self.conditions]
        roc_points.sort() 
        roc_points = [(0.0, 0.0)] + roc_points + [(1.0, 1.0)]#add 0.0 and 1.0 to complete the start and end of the curve
        auc = 0.0
        for i in range(1, len(roc_points)):
            x1, y1 = roc_points[i - 1]
            x2, y2 = roc_points[i]
            auc += (x2 - x1) * (y1 + y2) / 2
        return auc

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
