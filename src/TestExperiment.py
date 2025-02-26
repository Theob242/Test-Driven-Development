import unittest
from Experiment import Experiment
from SignalDetection import SignalDetection


class TestExperiment(unittest.TestCase):
    
    def setUp(self):
        self.exp = Experiment()
        self.sdt1 = SignalDetection(40, 10, 20, 30)  
        self.sdt2 = SignalDetection(30, 20, 10, 40)  
        self.sdt3 = SignalDetection(20, 30, 5, 45)   

    def test_add_condition(self):
        self.exp.add_condition(self.sdt1, "Condition A")
        self.assertEqual(len(self.exp.conditions), 1)

    def test_add_multiple_conditions(self): #Test adding multipe SDT conditions
        self.exp.add_condition(self.sdt1, "A")
        self.exp.add_condition(self.sdt2, "B")
        self.assertEqual(len(self.exp.conditions), 2)

    def test_sorted_roc_points(self): #Test ROC point return
        self.exp.add_condition(self.sdt1)
        self.exp.add_condition(self.sdt2)
        false_alarm_rates, hit_rates = self.exp.sorted_roc_points()
        self.assertTrue(all(false_alarm_rates[i] <= false_alarm_rates[i + 1] for i in range(len(false_alarm_rates) - 1)))

    def test_sorted_roc_points_empty_experiment(self): #Test if stored points raise ValueError
        with self.assertRaises(ValueError):
            self.exp.sorted_roc_points()

    def test_compute_auc(self): #test for AUC with better coordinates
        self.exp.add_condition(SignalDetection(0, 0, 0, 0))  # (0,0)
        self.exp.add_condition(SignalDetection(100, 0, 100, 0))  # (1,1)
        self.assertAlmostEqual(self.exp.compute_auc(), 0.5, places=2)

    def test_compute_auc_perfect_classification(self):
        self.exp.add_condition(SignalDetection(100, 0, 0, 100))  
        self.exp.add_condition(SignalDetection(100, 0, 100, 0))  
        self.assertAlmostEqual(self.exp.compute_auc(), 1.0, places=2)

    def test_compute_auc_empty_experiment(self):
        with self.assertRaises(ValueError):
            self.exp.compute_auc()

    def test_invalid_signal_detection_object(self): #test for typerror with invalid object
        with self.assertRaises(TypeError):
            self.exp.add_condition("Invalid Object", "Label")

if __name__ == "__main__":
    unittest.main()
