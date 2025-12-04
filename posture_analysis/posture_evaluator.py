from posture_analysis.leg_analyzer import LegAnalyzer
from posture_analysis.kyphosis_analyzer import KyphosisAnalyzer
from posture_analysis.lordosis_analyzer import LordosisAnalyzer

class PostureEvaluator:
    def __init__(self):
        self.leg_analyzer = LegAnalyzer()
        self.kyphosis_analyzer = KyphosisAnalyzer()
        self.lordosis_analyzer = LordosisAnalyzer()

    def analyze(self, landmarks, mask, image):
        leg_results = self.leg_analyzer.evaluate_leg_alignment(landmarks, mask, image)
        kyphosis_results = self.kyphosis_analyzer.analyze(landmarks, mask, image)
        lordosis_results = self.lordosis_analyzer.analyze(landmarks, mask, image)

        # Combine the results
        # For now, we'll just use the leg results for visualization
        # and combine the summaries.
        
        combined_summary = (
            f"--- Leg Analysis ---\n{leg_results['summary']}\n\n"
            f"--- Kyphosis Analysis ---\n{kyphosis_results['summary']}\n\n"
            f"--- Lordosis Analysis ---\n{lordosis_results['summary']}"
        )

        return {
            "summary": combined_summary,
            "visualized_image": leg_results["visualized_image"]
        }
