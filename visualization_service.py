import matplotlib.pyplot as plt
import numpy as np


class VisualizationService:
    def __init__(self, ndcg_scores, mae_scores, rmse_scores, no_of_fallback_predictions, user_fallback_movies, no_of_out_of_range_predicted_ratings_per_user, no_not_allowed_in_range_predicted_ratings, all_user_attempts, user_ids, unmatches, elapsed_time): #build_title):
        self.ndcg_scores = ndcg_scores
        self.mae_scores = mae_scores
        self.rmse_scores = rmse_scores
        self.no_of_fallback_predictions = no_of_fallback_predictions
        self.user_fallback_movies = user_fallback_movies
        self.no_of_out_of_range_predicted_ratings_per_user= no_of_out_of_range_predicted_ratings_per_user
        self.no_not_allowed_in_range_predicted_ratings = no_not_allowed_in_range_predicted_ratings
        self.all_user_attempts = all_user_attempts
        self.user_ids = user_ids
        self.unmatches = unmatches
        self.elapsed_time = elapsed_time
        #self.build_title = build_title  # function or lambda passed from UI

    def build_title(self, choice):
        def get_title(choice):
            return {
                1: "MAE & RMSE Distribution",
                2: "NDCG Distribution"
            }.get(choice, "Distribution")

        def format_fallback_movies():
            # Filter out ["no fallback movie"] entries
            filtered = [lst for lst in self.user_fallback_movies if lst != ["no fallback movie"]]
            if not filtered:
                return ""
            # Join movies within sublists with '\n' and separate sublists with double newline
            return "\n\n".join("\n".join(sublist) for sublist in filtered)

        def summarize_fallbacks():
            total = sum(self.no_of_fallback_predictions)
            fallback_users = [i for i, v in enumerate(self.no_of_fallback_predictions) if v > 0]
            lines = []
            if total > 0:
                lines.append(f"{total} fallback predictions")
            if fallback_users:
                lines.append(f"{len(fallback_users)} users with at least one fallback prediction")
                lines.append(f"List of users with fallback predictions: {fallback_users}")
            movies_str = format_fallback_movies()
            if movies_str.strip():
                lines.append(f"List of movies with fallback predictions:\n{movies_str}")
            return lines

        def summarize_rating_issues():
            lines = []
            total_out_of_range = sum(self.no_of_out_of_range_predicted_ratings_per_user)
            total_not_allowed = sum(self.no_not_allowed_in_range_predicted_ratings)
            if total_out_of_range > 0:
                lines.append(f"Total number of out of range predicted ratings: {total_out_of_range}")
            if total_not_allowed > 0:
                lines.append(f"Total number of not allowed within range predicted ratings: {total_not_allowed}")
            return lines

        def summarize_attempts():
            total_attempts = sum(self.all_user_attempts)
            lines = []
            if total_attempts > 0:
                lines.append(f"Total number of attempts across all users for rating prediction: {total_attempts}")

            non_zero_indices = [i for i, val in enumerate(self.all_user_attempts) if val > 0]
            non_zero_users = [self.user_ids[i] for i in non_zero_indices]

            if non_zero_users:
                lines.append(
                    f"Total number of users where attempts were involved for rating prediction: {len(non_zero_users)}")
                lines.append("List of users with at least one prediction attempt:")
                for i in range(0, len(non_zero_users), 10):
                    lines.append(", ".join(str(uid) for uid in non_zero_users[i:i + 10]))

            non_zero_items = [(self.user_ids[i], val) for i, val in enumerate(self.all_user_attempts) if val != 0]
            if non_zero_items:
                lines.append("List of attempts for each user by index:")
                for i in range(0, len(non_zero_items), 5):
                    lines.append(", ".join(str(item) for item in non_zero_items[i:i + 5]))

            return lines

        def summarize_reattempts():
            if not self.unmatches:
                return []
            reattempted = [a for a in self.unmatches if a > 0]
            if not reattempted:
                return []
            return [
                "List of movies reattempted for each user:",
                ", ".join(map(str, reattempted))
            ]

        # === Build summary ===
        title = get_title(choice)
        num_users = len(self.ndcg_scores)

        summary_lines = [f"{title} for {num_users} users"]
        summary_lines += summarize_fallbacks()
        summary_lines += summarize_rating_issues()
        summary_lines += summarize_attempts()
        summary_lines += summarize_reattempts()
        summary_lines.append(f"Processing time: {self.elapsed_time:.2f}s")

        return "\n".join(summary_lines)

    def plot_mae_rmse(self, ax):
        bins = np.linspace(
            min(min(self.mae_scores), min(self.rmse_scores)),
            max(max(self.mae_scores), max(self.rmse_scores)), 10
        )
        counts_mae, edges, _ = ax.hist(self.mae_scores, bins=bins, color='green', alpha=0.5, label='MAE',
                                       edgecolor='black')
        counts_rmse, _, _ = ax.hist(self.rmse_scores, bins=bins, color='red', alpha=0.5, label='RMSE',
                                    edgecolor='black')

        self._add_mean_std_lines(ax, self.mae_scores, 'green', 'MAE')
        self._add_mean_std_lines(ax, self.rmse_scores, 'red', 'RMSE')
        self._annotate_bins(ax, counts_mae, edges, 'green')
        self._annotate_bins(ax, counts_rmse, edges, 'red')

        ax.set_xticks(bins)
        ax.set_xticklabels([f"{tick:.2f}" for tick in bins], fontsize=10)
        ax.set_xlabel("Error Scores (MAE & RMSE)", fontsize=12)
        ax.set_ylabel("Number of Users", fontsize=12)
        ax.set_title(self.build_title(1), fontsize=14, multialignment='center')
        ax.legend(loc='best')

    def plot_ndcg(self, ax):
        bins = np.linspace(min(self.ndcg_scores), max(self.ndcg_scores), 10)
        counts, edges, _ = ax.hist(self.ndcg_scores, bins=bins, color='blue', alpha=0.5, label='NDCG',
                                   edgecolor='black')

        self._add_mean_std_lines(ax, self.ndcg_scores, 'blue', 'NDCG')
        self._annotate_bins(ax, counts, edges, 'black')

        ax.set_xticks(bins)
        ax.set_xticklabels([f"{tick:.2f}" for tick in bins], fontsize=10)
        ax.set_xlabel("NDCG Score", fontsize=12)
        ax.set_ylabel("Number of Users", fontsize=12)
        ax.set_title(self.build_title(2), fontsize=14, multialignment='center')
        ax.legend(loc='best')

    def save_plot(self, fig, filepath):
        fig.savefig(filepath, bbox_inches='tight', dpi=200)
        plt.close(fig)


    def _add_mean_std_lines(self, ax, scores, color, label):
        mean = np.mean(scores)
        std = np.std(scores)
        ax.axvline(mean, color=color, linestyle='--', linewidth=1.5, label=f'Mean {label}: {mean:.2f}')
        ax.axvline(min(scores), color=color, linestyle='--', linewidth=1.5, label=f'Min {label}: {min(scores):.2f}')
        ax.axvline(max(scores), color=color, linestyle='--', linewidth=1.5, label=f'Max {label}: {max(scores):.2f}')
        ax.axvspan(mean - std, mean + std, color=color, alpha=0.1, label=f'{label} ±1 Std: {std:.2f}')

        height = ax.get_ylim()[1]
        ax.text(mean - std, height * 0.5, f"{(mean - std):.2f}", ha='center', va='bottom', fontsize=10, color=color)
        ax.text(mean + std, height * 0.5, f"{(mean + std):.2f}", ha='center', va='bottom', fontsize=10, color=color)

    def _annotate_bins(self, ax, counts, bins, color):
        width = (bins[1] - bins[0]) / 2
        for count, edge in zip(counts, bins[:-1]):
            if count > 0:
                ax.text(edge + width, count + 0.5, str(int(count)), ha='center', va='bottom', fontsize=10, color=color)

    def update_figure(self, axes):
        """Update existing subplots with fresh visualizations."""
        axes[0].clear()
        axes[1].clear()
        self.plot_ndcg(axes[0])
        self.plot_mae_rmse(axes[1])
        plt.tight_layout()
