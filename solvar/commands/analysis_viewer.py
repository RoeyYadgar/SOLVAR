import os
import pickle
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any, Dict, List, Optional

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.path import Path as MplPath

from solvar.utils import sub_starfile


class AnalyzeViewer:
    """Interactive GUI for visualizing and analyzing coordinate data with UMAP and PCA projections.

    This class provides a tkinter-based interface for exploring coordinate data through
    various visualization modes including UMAP projections and principal component analysis.
    Users can interact with the plots to select cluster coordinates and save/load them.

    Attributes:
        master: The tkinter root window
        data: Dictionary containing coordinate data and analysis results
        dir: Directory path for file operations
        coords: Principal component coordinates
        umap_coords: UMAP-reduced coordinates
        cluster_coords: Selected cluster center coordinates
        umap_cluster_coords: UMAP coordinates of cluster centers
        selected_cluster_coords: List of selected cluster coordinates
        figures: Dictionary storing matplotlib figures
        color_by: StringVar for color coding selection
        figure_type: StringVar for figure type selection
    """

    def __init__(self, master: tk.Tk, data: Dict[str, Any], dir: str, max_points: int = None) -> None:
        """Initialize the AnalyzeViewer with data and directory.

        Args:
            master: The tkinter root window
            data: Dictionary containing coordinate data with keys 'coords', 'umap_coords',
                  and optionally 'cluster_coords' and 'umap_cluster_coords'
            dir: Directory path for file operations
        """
        self.master = master
        menubar = tk.Menu(self.master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self._open_new_file)
        filemenu.add_command(label="Load Star File", command=self._load_starfile)
        filemenu.add_command(label="Load latent coords", command=self._load_latent_coords)
        filemenu.add_command(label="Save Latent coords", command=self._save_cluster_coords)
        filemenu.add_command(label="Save Polygon Points", command=self._export_polygon_points)
        menubar.add_cascade(label="File", menu=filemenu)
        self.master.config(menu=menubar)
        self.data = data
        self.dir = dir
        self.coords = data["coords"]
        self.umap_coords = data["umap_coords"]
        self.cluster_coords = data.get("cluster_coords", None)
        self.umap_cluster_coords = data.get("umap_cluster_coords", None)
        self.selected_cluster_coords = []
        self.figures = {}
        self.color_by = tk.StringVar(value="None")
        self.figure_type = tk.StringVar(value="umap")
        self.mode = tk.StringVar(value="select")  # "select" or "polygon"
        self.polygon_vertices: List[List[float]] = []  # Store polygon vertices
        self.polygon_patch: Optional[PolygonPatch] = None  # Current polygon patch being drawn
        self.selected_indices_history: List[List[int]] = []  # History for undo
        self.polygon_selected_indices: Optional[np.ndarray] = None  # Indices of points inside completed polygon
        self.polygon_completed: bool = False  # Flag to track if polygon is completed
        self.polygon_figure_type: Optional[str] = None  # Figure type where polygon was created
        self.starfile_path: Optional[str] = None  # Path to loaded star file
        # Performance optimization: cache plot objects
        self._current_scatter = None  # Cache for scatter plot
        self._current_hexbin = None  # Cache for hexbin plot
        self._current_fig_type = None  # Current figure type ("scatter" or "hist")
        self._current_figure_type_key = None  # Current figure type key (e.g., "umap" or "pc_0_1")
        self._current_plot_coords = None  # Current coordinates being plotted
        self._max_points_for_scatter = max_points if max_points is not None else np.inf  # Downsample if more points
        # Pre-compute fixed downsampling indices for deterministic visualization
        n_total = len(self.coords)
        if n_total > self._max_points_for_scatter:
            np.random.seed(42)  # Fixed seed for reproducibility
            self._downsample_indices = np.random.choice(n_total, self._max_points_for_scatter, replace=False)
            np.random.seed()  # Reset seed to avoid affecting other random operations
        else:
            self._downsample_indices = None
        self._setup_gui()
        self._draw_figure()

    def _open_new_file(self) -> None:
        """Open a new analysis data file and update the viewer.

        Prompts user to select a pickle file containing analysis coordinates, loads the data, and
        refreshes the display.
        """
        path = filedialog.askopenfilename(
            title="Select analyze_coordinates pkl", filetypes=[("Pickle files", "*.pkl")], initialdir=self.dir
        )
        if not path:
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.dir = os.path.split(path)[0]
        self.data = data
        self.coords = data["coords"]
        self.umap_coords = data["umap_coords"]
        self.cluster_coords = data.get("cluster_coords", None)
        self.umap_cluster_coords = data.get("umap_cluster_coords", None)
        self.selected_cluster_coords = []
        self.polygon_vertices = []
        self.polygon_patch = None
        self.selected_indices_history = []
        self.polygon_selected_indices = None
        self.polygon_completed = False
        self.polygon_figure_type = None
        self.starfile_path = None
        # Reset performance caches
        self._current_scatter = None
        self._current_hexbin = None
        self._current_fig_type = None
        self._current_figure_type_key = None
        self._current_plot_coords = None
        # Recompute fixed downsampling indices for new dataset
        n_total = len(self.coords)
        if n_total > self._max_points_for_scatter:
            np.random.seed(42)  # Fixed seed for reproducibility
            self._downsample_indices = np.random.choice(n_total, self._max_points_for_scatter, replace=False)
            np.random.seed()  # Reset seed to avoid affecting other random operations
        else:
            self._downsample_indices = None
        self._draw_figure()

    def _setup_gui(self) -> None:
        """Set up the GUI components including controls and matplotlib canvas.

        Creates control frame with dropdown menus for figure type and color coding, buttons for
        cluster operations, and embeds a matplotlib figure for plotting.
        """
        # Top frame for controls (buttons and dropdowns)
        control_frame = ttk.Frame(self.master)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Dropdown for figure type
        figure_types = ["umap"]
        pc_dim = self.coords.shape[1]
        for i in range(min(5, pc_dim)):
            for j in range(i + 1, min(5, pc_dim)):
                figure_types.append(f"pc_{i}_{j}")
        ttk.Label(control_frame, text="Figure:").pack(side=tk.LEFT, padx=5)
        figure_menu = ttk.OptionMenu(
            control_frame,
            self.figure_type,
            self.figure_type.get(),
            *figure_types,
            command=lambda _: self._draw_figure(),
        )
        figure_menu.pack(side=tk.LEFT, padx=5)

        # Dropdown for color by
        color_options = ["None", "Density"] + [f"PC {i}" for i in range(self.coords.shape[1])]
        ttk.Label(control_frame, text="Color by:").pack(side=tk.LEFT, padx=5)
        color_menu = ttk.OptionMenu(
            control_frame, self.color_by, self.color_by.get(), *color_options, command=lambda _: self._draw_figure()
        )
        color_menu.pack(side=tk.LEFT, padx=5)

        # Mode selection
        ttk.Label(control_frame, text="Mode:").pack(side=tk.LEFT, padx=5)
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            mode_frame, text="Select", variable=self.mode, value="select", command=self._on_mode_change
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            mode_frame, text="Polygon", variable=self.mode, value="polygon", command=self._on_mode_change
        ).pack(side=tk.LEFT)

        # Buttons for cluster selection
        ttk.Button(control_frame, text="Reset Cluster Coords", command=self._reset_cluster_coords).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(control_frame, text="Undo", command=self._undo_last_selection).pack(side=tk.LEFT, padx=5)
        self.complete_polygon_btn = ttk.Button(
            control_frame, text="Complete Polygon", command=self._complete_polygon, state="disabled"
        )
        self.complete_polygon_btn.pack(side=tk.LEFT, padx=5)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _draw_figure(self) -> None:
        """Draw the current figure based on selected type and color scheme.

        Updates the matplotlib figure based on the current figure type (UMAP or PCA) and color
        coding selection. Handles both scatter plots and density histograms. Optimized to reuse
        existing plots and update data directly for better performance.
        """
        fig_type = self.figure_type.get()
        color_by = self.color_by.get()
        type_fig = "hist" if color_by == "Density" else "scatter"

        # Prepare labels for coloring
        if color_by == "None":
            # If there's a completed polygon, color points inside it red using labels
            if (
                self.polygon_selected_indices is not None
                and len(self.polygon_selected_indices) > 0
                and self.polygon_completed
            ):
                # Create labels array: 1 for points inside polygon, 0 for others
                # This will use colormap where 1=red and 0=gray/blue
                labels = np.zeros(len(self.coords))
                labels[self.polygon_selected_indices] = 1
            else:
                labels = None
        elif color_by == "Density":
            labels = None
        else:
            idx = int(color_by.split(" ")[1])
            labels = self.coords[:, idx]

        # Get coordinates to plot
        if fig_type == "umap":
            plot_coords = self.umap_coords
            cluster_coords_2d = self.umap_cluster_coords
            xlabel, ylabel = "UMAP 1", "UMAP 2"
        else:
            i, j = map(int, fig_type.split("_")[1:])
            plot_coords = self.coords[:, [i, j]]
            cluster_coords_2d = self.cluster_coords[:, [i, j]] if self.cluster_coords is not None else None
            xlabel, ylabel = f"PC {i}", f"PC {j}"

        # Check if we need to recreate the plot
        # Recreate if: plot type changed (scatter<->hist), figure type changed (umap<->pc), or first time
        plot_type_changed = self._current_fig_type != type_fig
        figure_type_key_changed = self._current_figure_type_key != fig_type
        is_first_time = self._current_plot_coords is None or self.ax is None

        needs_recreate = plot_type_changed or figure_type_key_changed or is_first_time

        if needs_recreate:
            # Clear the axes completely
            self.ax.clear()
            self._current_scatter = None
            self._current_hexbin = None
            self._current_fig_type = type_fig
            self._current_plot_coords = plot_coords.copy()
            # Reset polygon patch when axes are cleared so preview can be redrawn
            if self.polygon_patch is not None:
                self.polygon_patch = None

        # Downsample for very large datasets to improve performance (using fixed indices)
        if type_fig == "scatter" and self._downsample_indices is not None:
            # Use pre-computed fixed indices for deterministic visualization
            plot_coords_sampled = plot_coords[self._downsample_indices]
            labels_sampled = labels[self._downsample_indices] if labels is not None else None
        else:
            plot_coords_sampled = plot_coords
            labels_sampled = labels

        # Update or create the plot
        if type_fig == "scatter":
            if needs_recreate or self._current_scatter is None:
                # Remove hexbin if it exists
                if self._current_hexbin is not None:
                    self._current_hexbin.remove()
                    self._current_hexbin = None

                # Create new scatter plot
                self._current_scatter = self.ax.scatter(
                    plot_coords_sampled[:, 0],
                    plot_coords_sampled[:, 1],
                    s=0.1,
                    c=labels_sampled,
                    cmap="viridis" if labels_sampled is not None else None,
                    alpha=0.6,
                )
            else:
                # Update existing scatter plot data - much faster than recreating
                self._current_scatter.set_offsets(plot_coords_sampled)
                if labels_sampled is not None:
                    self._current_scatter.set_array(labels_sampled)
                    self._current_scatter.set_clim(vmin=labels_sampled.min(), vmax=labels_sampled.max())
                else:
                    self._current_scatter.set_array(None)
        else:  # hist/hexbin
            if needs_recreate or self._current_hexbin is None:
                # Remove scatter if it exists
                if self._current_scatter is not None:
                    self._current_scatter.remove()
                    self._current_scatter = None

                # Create hexbin plot using matplotlib hexbin directly
                hb = self.ax.hexbin(plot_coords[:, 0], plot_coords[:, 1], gridsize=50, cmap="Blues", mincnt=1)
                self._current_hexbin = hb
            # Note: hexbin plots are harder to update, so we recreate them when needed

        # Add cluster annotations
        if cluster_coords_2d is not None:
            # Remove old annotations
            for artist in list(self.ax.texts):
                if hasattr(artist, "_cluster_annotation"):
                    artist.remove()
            # Add new annotations
            for i in range(cluster_coords_2d.shape[0]):
                text = self.ax.annotate(str(i), (cluster_coords_2d[i, 0], cluster_coords_2d[i, 1]), fontweight="bold")
                text._cluster_annotation = True

        # Update axis labels and limits (only if figure type changed or first time)
        if needs_recreate:
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)

            # Set axis limits based on percentiles (excluding outliers)
            x_min, x_max = np.percentile(plot_coords[:, 0], [0.5, 99.5])
            x_delta = x_max - x_min
            y_min, y_max = np.percentile(plot_coords[:, 1], [0.5, 99.5])
            y_delta = y_max - y_min
            self.ax.set_xlim(x_min - 0.1 * x_delta, x_max + 0.1 * x_delta)
            self.ax.set_ylim(y_min - 0.1 * y_delta, y_max + 0.1 * y_delta)

        # Store current state for next update
        self._current_fig_type = type_fig
        self._current_figure_type_key = fig_type
        if needs_recreate:
            self._current_plot_coords = plot_coords.copy()

        # Redraw polygon only if in polygon mode, vertices exist, and figure type matches
        if (
            self.mode.get() == "polygon"
            and len(self.polygon_vertices) > 0
            and (not self.polygon_completed or self.polygon_figure_type == fig_type)
        ):
            self._draw_polygon()

        # Update complete polygon button state
        self._update_polygon_button_state()
        self.canvas.draw()

    def _on_mode_change(self) -> None:
        """Handle mode change event."""
        if self.mode.get() == "select":
            # Clear polygon when switching to select mode
            self.polygon_vertices = []
            self.polygon_patch = None
            self.polygon_completed = False
            self.polygon_figure_type = None
        else:
            # Reset completion state when entering polygon mode
            self.polygon_completed = False
            self.polygon_figure_type = None
        self._update_polygon_button_state()
        self._draw_figure()

    def _update_polygon_button_state(self) -> None:
        """Update the state of the Complete Polygon button based on polygon vertices."""
        if hasattr(self, "complete_polygon_btn"):
            if self.mode.get() == "polygon" and len(self.polygon_vertices) >= 3 and not self.polygon_completed:
                self.complete_polygon_btn.config(state="normal")
            else:
                self.complete_polygon_btn.config(state="disabled")

    def _on_motion(self, event: Any) -> None:
        """Handle mouse motion events for drawing polygon preview."""
        # Stop preview if polygon is completed
        if self.polygon_completed:
            return
        if self.mode.get() != "polygon" or len(self.polygon_vertices) == 0:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Draw preview of polygon with current mouse position
        if self.ax is not None:
            # Remove old preview
            if self.polygon_patch is not None:
                self.polygon_patch.remove()

            # Create preview with current mouse position
            preview_vertices = self.polygon_vertices + [[event.xdata, event.ydata]]
            if len(preview_vertices) >= 2:
                self.polygon_patch = PolygonPatch(
                    preview_vertices, fill=False, edgecolor="blue", linestyle="--", linewidth=2, alpha=0.5
                )
                self.ax.add_patch(self.polygon_patch)
                self.canvas.draw_idle()

    def _on_click(self, event: Any) -> None:
        """Handle mouse click events on the plot.

        In select mode: finds the nearest data point and adds it to cluster coordinates.
        In polygon mode: adds vertex to polygon, or completes polygon on double-click.

        Args:
            event: Matplotlib mouse event containing click coordinates
        """
        # Only respond to clicks inside the axes
        if event.xdata is None or event.ydata is None:
            return

        if self.mode.get() == "select":
            self._on_click_select(event)
        elif self.mode.get() == "polygon":
            self._on_click_polygon(event)

    def _on_click_select(self, event: Any) -> None:
        """Handle click in select mode to add cluster coordinate."""
        fig_type = self.figure_type.get()
        # Get the current visible coordinates
        if fig_type == "umap":
            coords = self.umap_coords
        else:
            i, j = map(int, fig_type.split("_")[1:])
            coords = self.coords[:, [i, j]]

        # Find the closest point in coords to the click
        click_point = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(coords - click_point, axis=1)
        idx = np.argmin(dists)

        # Save state for undo
        if self.umap_cluster_coords is not None:
            self.selected_indices_history.append([len(self.umap_cluster_coords)])
        else:
            self.selected_indices_history.append([0])

        # vstack to umap_cluster_coords or cluster_coords
        if self.umap_cluster_coords is None:
            self.umap_cluster_coords = np.array([self.umap_coords[idx]])
            self.cluster_coords = np.array([self.coords[idx]])
        else:
            self.umap_cluster_coords = np.vstack([self.umap_cluster_coords, self.umap_coords[idx][None, :]])
            self.cluster_coords = np.vstack([self.cluster_coords, self.coords[idx][None, :]])

        self._draw_figure()

    def _on_click_polygon(self, event: Any) -> None:
        """Handle click in polygon mode to add vertex or complete polygon."""
        # Don't add vertices if polygon is already completed
        if self.polygon_completed:
            return
        # Right-click completes the polygon
        if event.button == 3 and len(self.polygon_vertices) >= 3:
            self._complete_polygon()
        elif event.button == 1:  # Left click adds a vertex
            # Single click adds a vertex
            self.polygon_vertices.append([event.xdata, event.ydata])
            self._update_polygon_button_state()
            self._draw_figure()

    def _on_button_release(self, event: Any) -> None:
        """Handle button release events (used for tracking mouse state)."""
        # This can be used for additional tracking if needed
        pass

    def _draw_polygon(self) -> None:
        """Draw the current polygon on the plot."""
        if self.ax is None or len(self.polygon_vertices) < 2:
            return

        # Remove old polygon if exists
        if self.polygon_patch is not None:
            self.polygon_patch.remove()

        # Draw the polygon
        self.polygon_patch = PolygonPatch(self.polygon_vertices, fill=False, edgecolor="red", linewidth=2, alpha=0.8)
        self.ax.add_patch(self.polygon_patch)

    def _complete_polygon(self) -> None:
        """Complete polygon and save indices of points inside it."""
        if len(self.polygon_vertices) < 3:
            return

        # Get current visible coordinates
        fig_type = self.figure_type.get()
        if fig_type == "umap":
            coords_2d = self.umap_coords
        else:
            i, j = map(int, fig_type.split("_")[1:])
            coords_2d = self.coords[:, [i, j]]

        # Create path from polygon vertices
        polygon_path = MplPath(self.polygon_vertices)

        # Find points inside polygon and save indices
        inside_mask = polygon_path.contains_points(coords_2d)
        self.polygon_selected_indices = np.where(inside_mask)[0]

        # Mark polygon as completed and save the figure type it was created in
        self.polygon_completed = True
        self.polygon_figure_type = fig_type

        # Clear preview patch and draw final polygon
        if self.polygon_patch is not None:
            self.polygon_patch.remove()
            self.polygon_patch = None
        self._draw_polygon()

        # Disable button and stop allowing new vertices
        self._update_polygon_button_state()
        self._draw_figure()

    def _undo_last_selection(self) -> None:
        """Undo the last selection action."""
        if not self.selected_indices_history or self.umap_cluster_coords is None:
            return

        # Get indices to remove from last action
        indices_to_remove = self.selected_indices_history.pop()
        if not indices_to_remove:
            return

        # Remove those indices (they should be the last ones added)
        if len(indices_to_remove) == len(self.umap_cluster_coords):
            # All points were added in last action
            self.umap_cluster_coords = None
            self.cluster_coords = None
        else:
            # Remove only the last added points
            keep_indices = [i for i in range(len(self.umap_cluster_coords)) if i not in indices_to_remove]
            if keep_indices:
                self.umap_cluster_coords = self.umap_cluster_coords[keep_indices]
                self.cluster_coords = self.cluster_coords[keep_indices]
            else:
                self.umap_cluster_coords = None
                self.cluster_coords = None

        self._draw_figure()

    def _load_starfile(self) -> None:
        """Load a star file for exporting polygon-selected particles."""
        path = filedialog.askopenfilename(
            title="Select Star File", filetypes=[("Star files", "*.star")], initialdir=self.dir
        )
        if path:
            self.starfile_path = path
            tk.messagebox.showinfo("Star File Loaded", f"Star file loaded: {os.path.basename(path)}")

    def _export_polygon_points(self) -> None:
        """Export indices of points that are inside the completed polygon."""
        # Check if there are saved polygon indices
        if self.polygon_selected_indices is None or len(self.polygon_selected_indices) == 0:
            # Check if there's an active polygon that hasn't been completed yet
            if len(self.polygon_vertices) >= 3:
                # Complete the polygon first
                self._complete_polygon()
            else:
                tk.messagebox.showwarning("No Points", "No polygon completed. Please complete a polygon first.")
                return

        if self.polygon_selected_indices is None or len(self.polygon_selected_indices) == 0:
            tk.messagebox.showwarning("No Points", "No points inside the polygon.")
            return

        # Check if star file is loaded and offer to export as starfile
        if self.starfile_path is not None:
            # Ask user if they want to export as starfile or pickle
            choice = tk.messagebox.askyesnocancel(
                "Export Format",
                "Star file is loaded. Export as starfile (Yes) or pickle (No)?\n" "Click Cancel to abort.",
            )
            if choice is None:  # Cancel
                return
            elif choice:  # Yes - export as starfile
                # Ask for output starfile path
                starfile_path = filedialog.asksaveasfilename(
                    defaultextension=".star",
                    filetypes=[("Star files", "*.star")],
                    initialdir=self.dir,
                    initialfile="selected_particles.star",
                )
                if starfile_path:
                    try:
                        # Convert to list if numpy array for compatibility
                        indices = (
                            self.polygon_selected_indices.tolist()
                            if isinstance(self.polygon_selected_indices, np.ndarray)
                            else self.polygon_selected_indices
                        )
                        sub_starfile(self.starfile_path, starfile_path, indices)
                        tk.messagebox.showinfo(
                            "Success", f"Star file exported to:\n{starfile_path}\n\n{len(indices)} particles selected."
                        )
                    except Exception as e:
                        tk.messagebox.showerror("Error", f"Failed to export starfile:\n{str(e)}")
                return

        # Export as pickle file
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")],
            initialdir=self.dir,
            initialfile="ind.pkl",
        )
        if path:
            with open(path, "wb") as f:
                pickle.dump(self.polygon_selected_indices, f)

    def _reset_cluster_coords(self) -> None:
        """Reset all selected cluster coordinates and refresh the display.

        Clears both UMAP and PCA cluster coordinate arrays and redraws the figure.
        """
        self.umap_cluster_coords = None
        self.cluster_coords = None
        self.selected_indices_history = []
        self.polygon_vertices = []
        self.polygon_patch = None
        self.polygon_selected_indices = None
        self.polygon_completed = False
        self.polygon_figure_type = None
        self.starfile_path = None
        # Reset performance caches
        self._current_scatter = None
        self._current_hexbin = None
        self._current_fig_type = None
        self._current_figure_type_key = None
        self._current_plot_coords = None
        self._update_polygon_button_state()
        self._draw_figure()

    def _save_cluster_coords(self) -> None:
        """Save the current cluster coordinates to a pickle file.

        Prompts user to select a save location and saves the cluster coordinates as a pickle file
        for later use.
        """
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")],
            initialdir=self.dir,
            initialfile="latent_coords.pkl",
        )
        if path:
            with open(path, "wb") as f:
                pickle.dump(self.cluster_coords, f)

    def _load_latent_coords(self) -> None:
        """Load cluster coordinates from a pickle file.

        Prompts user to select a pickle file containing cluster coordinates, loads them, and maps
        them to UMAP coordinates for display.
        """
        path = filedialog.askopenfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")],
            initialdir=self.dir,
            initialfile="latent_coords.pkl",
        )
        if path:
            with open(path, "rb") as f:
                self.cluster_coords = pickle.load(f)

            self.umap_cluster_coords = np.zeros((len(self.cluster_coords), self.umap_coords.shape[1]))
            for i, cluster_center in enumerate(self.cluster_coords):
                matches = np.where(np.all(self.coords == cluster_center, axis=1))[0]
                if len(matches) > 0:
                    idx = matches[0]
                else:
                    print(
                        "Loaded latent coords do not match exact points from "
                        "existing latent points. Displaying closest points instead"
                    )
                    # Find the closest coordinate
                    dists = np.linalg.norm(self.coords - cluster_center, axis=1)
                    idx = np.argmin(dists)
                self.umap_cluster_coords[i] = self.umap_coords[idx]
            self._draw_figure()

    def _on_close(self) -> None:
        """Handle window close event.

        Properly closes the tkinter window and quits the application.
        """
        self.master.quit()
        self.master.destroy()


class NumberWithSuffix(click.ParamType):
    """Click parameter type for parsing numbers with suffix (e.g., 10k, 2M, 500)."""

    name = "number_with_suffix"

    def convert(self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        s = str(value).strip().lower()
        if s.endswith("k"):
            return int(float(s[:-1]) * 1_000)
        if s.endswith("m"):
            return int(float(s[:-1]) * 1_000_000)
        try:
            return int(s)
        except ValueError:
            self.fail(f"{value!r} is not a valid number (e.g., 10k, 2M, 500)", param, ctx)


NUMBER_WITH_SUFFIX = NumberWithSuffix()


def read_cryodrgn_format(analysis_dir: str) -> Dict:
    data_dict = {}
    output_dir, analysis_dirname = os.path.split(analysis_dir)

    epoch_num = os.path.splitext(analysis_dirname)[1].replace(".", "")

    coords_path = os.path.join(output_dir, f"z.{epoch_num}.pkl")

    with open(coords_path, "rb") as f:
        data_dict["coords"] = pickle.load(f)

    umap_path = os.path.join(analysis_dir, "umap.pkl")

    with open(umap_path, "rb") as f:
        data_dict["umap_coords"] = pickle.load(f)

    return data_dict


@click.command()
@click.argument("path", required=False, type=str)
@click.option(
    "-n",
    "--max-points",
    type=NUMBER_WITH_SUFFIX,
    default=None,
    help="Max points to display (e.g., 10k, 2M, 500). Used for performance optimization with large datasets.",
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["solvar", "cryodrgn"], case_sensitive=False),
    default="solvar",
    help="Input data format (default: solvar)",
)
def analysis_viewer_cli(path: Optional[str], max_points: Optional[int], format: str) -> None:
    """Launch the interactive analysis viewer GUI.

    Opens an interactive GUI for visualizing and analyzing coordinate data with
    UMAP and PCA projections. Users can select cluster coordinates and save/load them.

    PATH: Path to analyze_coordinates pkl file (optional, will prompt if not provided)
    """
    root = tk.Tk()
    root.title("Analyze Coordinates Viewer")

    if path:
        file_path = path
    else:
        file_path = filedialog.askopenfilename(
            title="Select analyze_coordinates pkl", filetypes=[("Pickle files", "*.pkl")]
        )
        if not file_path:
            print("No file selected.")
            return

    if format.lower() == "solvar":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    elif format.lower() == "cryodrgn":
        data = read_cryodrgn_format(file_path)
    else:
        raise ValueError(f"Unknown format: {format}")

    AnalyzeViewer(root, data, dir=os.path.split(file_path)[0], max_points=max_points)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()


def main() -> None:
    """Main function to launch the AnalyzeViewer application.

    This is kept for backward compatibility. Use analysis_viewer_cli() for the Click interface.
    """
    analysis_viewer_cli()


if __name__ == "__main__":
    main()
