from dataclasses import dataclass


@dataclass
class FusionOutput:
    state: str
    score_visual: float
    prob_target: float
    prob_noise: float
    score_final: float


class ThreeStateMachine:
    def __init__(self, conf_thres: float, fusion_thres: float, lambda_rf: float, alpha_noise: float):
        self.conf_thres = conf_thres
        self.fusion_thres = fusion_thres
        self.lambda_rf = lambda_rf
        self.alpha_noise = alpha_noise

    def score(self, score_visual: float, prob_target: float, prob_noise: float):
        raw = (score_visual + self.lambda_rf * prob_target) * ((1.0 - prob_noise) ** self.alpha_noise)
        return max(0.0, min(1.0, raw))

    def decide(self, score_visual: float, prob_target: float, prob_noise: float, is_target: bool):
        score_final = self.score(score_visual=score_visual, prob_target=prob_target, prob_noise=prob_noise)
        state = "State_III_NLOS_Rescue"
        if is_target and score_visual > self.conf_thres and score_final > self.fusion_thres and prob_target > 0.7:
            state = "State_I_HighConfidence_Confirmation"
        elif (not is_target) and score_visual > self.conf_thres and score_final < self.fusion_thres and prob_target < 0.3:
            state = "State_II_FalseAlarm_Suppression"
        return FusionOutput(
            state=state,
            score_visual=float(score_visual),
            prob_target=float(prob_target),
            prob_noise=float(prob_noise),
            score_final=float(score_final),
        )
