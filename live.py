import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, Video, display
from html import escape


class LivePlot:
    def __init__(self, rl=False):
        self.rl = rl
        self.figure, self.axes = plt.subplots(2, 4, figsize=(16, 6))
        plt.close(self.figure)
        self.handle = None
        self.video_handle = None

    def __call__(self, history, diagnostics, title, episodes=None):
        history, diagnostics = pd.DataFrame(history), pd.DataFrame(diagnostics)
        if self.rl:
            episodes = pd.DataFrame(episodes)
            if not episodes.empty:
                episodes = episodes.assign(reward=episodes.reward.rolling(20, min_periods=1).mean())
            panels = [
                (history, "step", ["reward_mean"], "Evaluation reward ± episode SD"),
                (episodes, "step", ["reward"], "Training reward, last 20 episodes"),
                (diagnostics, "step", ["td_loss"], "TD loss"),
                (diagnostics, "step", ["epsilon"], "Exploration epsilon"),
            ]
        else:
            panels = [
                (history, "epoch", ["train_eval_loss", "validation_loss"], "Loss, eval mode"),
                (history, "epoch", ["train_accuracy", "validation_accuracy"], "Accuracy, %"),
                (history, "epoch", ["gap"], "Validation − train loss"),
                (history, "epoch", ["train_loss"], "Training loss"),
            ]
        panels += [(diagnostics, "step", [column], title) for column, title in [
            ("lr", "Learning rate"), ("weight_l2", "Weight L2 norm"),
            ("grad_l2_clipped" if self.rl else "grad_l2", "Gradient L2 norm (clipped)" if self.rl else "Gradient L2 norm"),
            ("update_ratio", "Relative weight update"),
        ]]
        for axis, (frame, x, columns, label) in zip(self.axes.flat, panels):
            axis.clear()
            for column in columns:
                if column in frame:
                    axis.plot(frame[x], frame[column], label=column)
            if "reward_mean" in columns and "reward_std" in frame:
                axis.fill_between(frame[x], frame.reward_mean - frame.reward_std,
                                  frame.reward_mean + frame.reward_std, alpha=0.15)
            if len(columns) > 1 and not frame.empty:
                axis.legend(fontsize=8)
            axis.set(title=label, xlabel=x)
            axis.grid(alpha=0.2)
        self.figure.suptitle(title)
        self.figure.tight_layout(rect=(0, 0, 1, 0.95))
        if self.handle is None:
            self.handle = display(self.figure, display_id=True)
        else:
            self.handle.update(self.figure)

    def video(self, path, title):
        content = HTML(f"<p>{escape(title)}</p>" + Video(str(path), embed=True, width=600)._repr_html_())
        if self.video_handle is None:
            self.video_handle = display(content, display_id=True)
        else:
            self.video_handle.update(content)
