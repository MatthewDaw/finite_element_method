"""Class for visualizing meshes and solutions."""

import matplotlib.pyplot as plt

class Visualizer:
    """Class for visualizing meshes and solutions."""

    def show_points_and_their_values(self, points, u=None, points_of_interest=None):
        """Show points."""

        plt.figure(figsize=(8, 6))
        if u is not None:
            scatter = plt.scatter(points[:, 0], points[:, 1], c=u, cmap='viridis', s=100, edgecolor='k')
            # Add a colorbar to show value mapping
            cbar = plt.colorbar(scatter)
            cbar.set_label('Value')
        else:
            plt.scatter(points[:, 0], points[:, 1], s=100, edgecolor='k')

        if points_of_interest is not None:
            plt.scatter(points_of_interest[:, 0], points_of_interest[:, 1], color='black', s=150, marker='X',
                        label="Points of Interest")

        # Add labels and title
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title('Points Visualization Colored by Value')
        plt.grid(True)
        plt.show()

