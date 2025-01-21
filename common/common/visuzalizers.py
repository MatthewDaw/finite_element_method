"""Class for visualizing meshes and solutions."""

import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon

class Visualizer:
    """Class for visualizing meshes and solutions."""

    def show_points_and_their_values(self, points, u=None, points_of_interest=None, shapely_polygon=None):
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

        if shapely_polygon is not None:
            plot_polygon(shapely_polygon)

        # Add labels and title
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title('Points Visualization Colored by Value')
        plt.grid(True)
        plt.show()

    def display_mesh(self, p, t=None, e=None, u=None):
        """
        Visualize a triangular mesh with optional highlighted edges and point coloring.

        Parameters:
            p (ndarray): Nx2 array of mesh points.
            t (ndarray): Mx3 array of triangles (vertex indices).
            e (ndarray): Optional Kx2 array of edges to highlight (vertex indices).
            u (ndarray): Optional N-array of values for mesh points, used for coloring.
        """
        plt.figure(figsize=(8, 6))

        if t is not None:
            # Plot the triangles
            plt.triplot(p[:, 0], p[:, 1], t[:, :3], color="blue", linewidth=0.8, alpha=0.8, label="Triangles")
        else:
            plt.scatter(p[:, 0], p[:, 1], color="blue", s=20, zorder=5, label="Points")

        # Color the points based on their values if `u` is provided
        if u is not None:
            scatter = plt.scatter(
                p[:, 0],
                p[:, 1],
                c=u,
                cmap="viridis",
                s=40,
                zorder=5,
                label="Point Values",
            )
            plt.colorbar(scatter, label="Point Values")
        else:
            # Default scatter if `u` is not provided
            plt.scatter(p[:, 0], p[:, 1], color="red", s=20, zorder=5, label="Points")

        # Highlight specific edges if provided
        if e is not None:
            for edge in e[:,:2]:
                plt.plot(
                    p[edge, 0],
                    p[edge, 1],
                    color="orange",
                    linewidth=2,
                    label="Highlighted Edge" if "Highlighted Edge" not in plt.gca().get_legend_handles_labels()[
                        1] else "",
                )

        # Set plot attributes
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Mesh Visualization", fontsize=14)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        # Show the plot
        plt.show()

