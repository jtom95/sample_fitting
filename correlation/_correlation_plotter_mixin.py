from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CorrelationPlotterMixin:
    def plot_combined_correlations(
        self,
        cmap: str = "hot",
        n_per_row: int = 3,
        figsize: tuple = (6, 6),
        aspect="auto",
        apply_variable_alpha: bool = False,
        use_position_grid: bool = True,
        include_colorbars: bool = False,
        show_axes_labels: bool = False,
        show_axes_ticks: bool = True,
        text_space_ratio: float = 0.1,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot the combined correlation matrices for each frequency index.

        Args:
            cmap (str, optional): The colormap to use for the plots. Defaults to "hot".
            n_per_row (int, optional): The number of plots per row. Defaults to 3.
            figsize (tuple, optional): The figure size. Defaults to (6, 6).
            aspect (str, optional): The aspect ratio of the plots. Defaults to "auto".
            apply_variable_alpha (bool, optional): Whether to apply variable alpha based on the mean matrix. Defaults to False.
            use_position_grid (bool, optional): Whether to use the position grid for axis labels. Defaults to True.
            include_colorbars (bool, optional): Whether to include colorbars for each plot. Defaults to False.
            **kwargs: Additional keyword arguments to pass to imshow.

        Returns:
            Tuple[plt.Figure, np.ndarray]: The figure containing the plots and the array of axes.
        """
        n_plots = len(self.frequency_indices)
        n_rows = (n_plots - 1) // n_per_row + 1
        n_cols = n_per_row if n_plots > n_per_row else n_plots

        width_ratios = [text_space_ratio] + [1] * n_cols
        height_ratios = [text_space_ratio] + [1] * n_rows

        fig, axs = plt.subplots(n_rows+1, n_cols+1, figsize=figsize, width_ratios=width_ratios, height_ratios=height_ratios, constrained_layout=True)
        axs_top = axs[0, :]
        axs_left= axs[:, 0]

        for axx in axs_top:
            axx.axis("off")
        for axx in axs_left:
            axx.axis("off")

        axs = axs[1:, 1:]
        axs_flat = axs.flatten()

        returned_axes = np.empty((n_rows, n_cols), dtype=object)

        for i, freq_index in enumerate(self.frequency_indices):
            ax = axs_flat[i]
            fig0, axes = self.plot_combined_correlation_at_index(
                index=freq_index,
                ax=ax,
                cmap=cmap,
                apply_variable_alpha=apply_variable_alpha,
                aspect=aspect,
                use_position_grid=use_position_grid,
                **kwargs,
            )
            returned_axes[i // n_cols, i % n_cols] = axes

            if hasattr(self, "critical_frequencies"):
                ax.set_title(
                    f"{self.critical_frequencies[i]/1e6:.1f}MHz [{freq_index}]"
                )
            else: 
                ax.set_title(
                    f"[{freq_index}]"
                )
            if not include_colorbars:
                ax.images[-1].colorbar.remove()

        # Remove unused subplots
        for i in range(n_plots, len(axs_flat)):
            axs_flat[i].axis("off")
            axs_flat[i].twiny().axis("off")
            axs_flat[i].twinx().axis("off")

        # Remove labels and ticks from inner plots
        for i in range(n_rows):
            for j in range(n_cols):
                if i + j >= n_plots:
                    break
                tuple_axis = returned_axes[i, j]
                if tuple_axis is None:
                    continue
                ax = tuple_axis[0]
                ax_right = tuple_axis[1]
                ax_top = tuple_axis[2]

                if i < n_rows - 1:
                    ax.set_xlabel("")
                    # ax.set_xticks([])

                if j > 0:
                    ax.set_ylabel("")
                    # ax.set_yticks([])

                if i > 0:
                    ax_top.set_xlabel("")
                    # ax_top.set_xticks([])
                if j < n_cols - 1:
                    ax_right.set_ylabel("")
                    # ax_right.set_yticks([])

        if not show_axes_labels:
            for tuple_ax in returned_axes.flatten():
                if tuple_ax is not None:
                    ax = tuple_ax[0]
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax_top = tuple_ax[2]
                    ax_right = tuple_ax[1]
                    ax_top.set_xlabel("")
                    ax_right.set_ylabel("")
        if not show_axes_ticks:
            for tuple_ax in returned_axes.flatten():
                if tuple_ax is not None:
                    ax = tuple_ax[0]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax_top = tuple_ax[2]
                    ax_right = tuple_ax[1]
                    ax_top.set_xticks([])
                    ax_right.set_yticks([])   

        # draw the figure to get the positions
        fig.canvas.draw()
        # left text
        axs_left: List[plt.Axes]
        p1 = axs_left[0].get_position(False)
        p2 = axs_left[-1].get_position(False)

        x_mean = (p1.x0 + p2.x1) / 2
        y_mean = (p1.y0 + p2.y1) / 2

        fig.text(x_mean, 0.5, "X Correlation", ha='center', va='center', fontsize=12, rotation='vertical')

        # top text
        q1 = axs_top[0].get_position(True)
        q2 = axs_top[-1].get_position(True)

        x_mean = (q1.x0 + q2.x1) / 2
        y_mean = (q1.y0 + q2.y1) / 2

        fig.text(0.5, y_mean, "Y Correlation", ha='center', va='center', fontsize=12)        
        return fig, axs

    def plot_combined_correlation_at_index(
        self,
        index=None,
        ax=None,
        cmap: str = "hot",
        apply_variable_alpha: bool = True,
        aspect="auto",
        use_position_grid: bool = False,
        n_twin_labels: int = 3,
        **kwargs,
    ):
        """
        Plot the combined correlation matrix at the specified index.

        Args:
            index (int, optional): The frequency index. Defaults to None (first index).
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None (creates a new figure).
            cmap (str, optional): The colormap to use for the plot. Defaults to "hot".
            apply_variable_alpha (bool, optional): Whether to apply variable alpha based on the mean matrix. Defaults to True.
            aspect (str, optional): The aspect ratio of the plot. Defaults to "auto".
            use_position_grid (bool, optional): Whether to use the position grid for axis labels. Defaults to False.
            **kwargs: Additional keyword arguments to pass to imshow.
        """
        if aspect != "auto":
            raise ValueError("Aspect ratio must be 'auto' for combined correlation matrix.")
        index_position = self.frequency_indices_order_dict.get(index, None)
        if index_position is None:
            raise ValueError("The frequency index is not in the list of frequency indices.")
        combined_corr = self.combined_correlation_matrix[..., index_position]

        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
        else: 
            fig = ax.figure

        ax: plt.Axes

        if apply_variable_alpha:
            mean_matrix_x = self.x_mean_matrix[..., index_position]
            mean_matrix_y = self.y_mean_matrix[..., index_position]
            mean_matrix = np.triu(mean_matrix_y) + np.tril(mean_matrix_x)
            np.fill_diagonal(mean_matrix, 0.5 * (np.diag(mean_matrix_x) + np.diag(mean_matrix_y)))
            alpha = (mean_matrix - mean_matrix.min()) / (mean_matrix.max() - mean_matrix.min())
        else:
            alpha = 1

        x_dim = self.x_correlation_matrix.shape[0]
        y_dim = self.y_correlation_matrix.shape[1]
        xaxis = self.position_grid.x
        yaxis = self.position_grid.y

        if use_position_grid:
            if x_dim < y_dim:
                extra_steps = y_dim - x_dim
                x_step = np.mean(np.diff(xaxis))
                x_corr_limits = (xaxis[0], xaxis[-1]+extra_steps*x_step)
                y_corr_limits = (yaxis[0], yaxis[-1])
            elif x_dim > y_dim:
                extra_steps = x_dim - y_dim
                y_step = np.mean(np.diff(yaxis))
                y_corr_limits = (yaxis[0], yaxis[-1]+extra_steps*y_step)
                x_corr_limits = (xaxis[0], xaxis[-1])
            else:
                x_corr_limits = (xaxis[0], xaxis[-1])
                y_corr_limits = (yaxis[0], yaxis[-1])
        else:
            y_corr_limits = (0, combined_corr.shape[0])
            x_corr_limits = (0, combined_corr.shape[0]) # the correlation matrix is square

        if self.position_grid is None:
            raise ValueError("Position grid is not available.")
        extent = [
            x_corr_limits[0],
            x_corr_limits[1],
            x_corr_limits[0],
            x_corr_limits[1],
        ]

        im = ax.imshow(combined_corr, extent=extent, cmap=cmap, alpha=alpha, aspect=aspect, origin="lower", vmin=0, vmax=1, **kwargs)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Combined Correlation")

        if use_position_grid:
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")
            ax.yaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")

        # Add extra x-label and y-label for Y Position
        ax_top = ax.twiny()
        ax_right = ax.twinx()

        if use_position_grid:
            ax_right.invert_yaxis()
            # Set the tick locations and labels for the top and right axes
            # chose 5 evenly spaced ticks
            y_values_espaced = np.linspace(yaxis.min(), yaxis.max(), n_twin_labels)
            ax_top.set_xticks(y_values_espaced)
            ax_top.set_xticklabels([f"{y*1e3:.0f}" for y in y_values_espaced])
            ax_top.set_xticks(np.linspace(yaxis.min(), yaxis.max(), n_twin_labels*5), minor=True)
            ax_top.set_xlim(y_corr_limits)

            ax_right.set_yticks(y_values_espaced)
            ax_right.set_yticklabels([f"{y*1e3:.0f}" for y in y_values_espaced])
            ax_right.set_yticks(np.linspace(yaxis.min(), yaxis.max(), n_twin_labels*5), minor=True)
            ax_right.set_ylim(y_corr_limits[1], y_corr_limits[0])

            # Chose 5 evenly spaced ticks
            x_values_espaced = np.linspace(xaxis.min(), xaxis.max(), n_twin_labels)
            ax.set_xticks(x_values_espaced)
            ax.set_xticklabels([f"{x*1e3:.0f}" for x in x_values_espaced])
            ax.set_xticks(np.linspace(xaxis.min(), xaxis.max(), n_twin_labels*5), minor=True)
            ax.set_xlim(x_corr_limits)

            ax.set_yticks(x_values_espaced)
            ax.set_yticklabels([f"{x*1e3:.0f}" for x in x_values_espaced])
            ax.set_yticks(np.linspace(xaxis.min(), xaxis.max(), n_twin_labels*5), minor=True)           
            ax.set_ylim(x_corr_limits)
            ax.invert_yaxis()
            # Invert the top and right axes

            # shift the spines and set the labels
            ax_right.spines["right"].set_position(
                ("axes", 1 - (y_corr_limits[-1]-yaxis[-1]) / (y_corr_limits[-1] - y_corr_limits[0]))
            )
            # shift the x axes labels to the bottom to where y_right is equal to yaxis[0]
            ax.spines["bottom"].set_position(
                (
                    "axes",
                    (x_corr_limits[-1] - xaxis[-1]) / (x_corr_limits[-1] - x_corr_limits[0])
                    # - x_step /2 / (x_corr_limits[-1] - x_corr_limits[0]),
                )
            )

            # write the labels in the correct place
            desired_xlabel_value = (xaxis[-1] + xaxis[0]) / 2
            transformed_x_value_x, _ = ax.transAxes.inverted().transform(
                ax.transData.transform((desired_xlabel_value, 0))
            )
            _, transformed_x_value_y = ax.transAxes.inverted().transform(
                ax.transData.transform((0, desired_xlabel_value))
            )

            ax.set_xlabel("X Position [mm]", ha="center", va="top", x=transformed_x_value_x)
            ax.set_ylabel("X Position [mm]", ha="center", va="bottom", y=transformed_x_value_y)

            # write the labels in the correct place
            desired_ylabel_value = (yaxis[-1] + yaxis[0]) / 2
            transformed_y_value_x, _ = ax_top.transAxes.inverted().transform(
                ax_top.transData.transform((desired_ylabel_value, 0))
            )
            _, transformed_y_value_y = ax_right.transAxes.inverted().transform(
                ax_right.transData.transform((0, desired_ylabel_value))
            )

            ax_top.set_xlabel("Y Position [mm]", ha="center", va="bottom", x=transformed_y_value_x)
            ax_right.set_ylabel("Y Position [mm]", ha="center", va="top", y=transformed_y_value_y)
            
        else:

            ax_top.set_xlim(0, combined_corr.shape[0])
            ax_right.set_ylim(0, combined_corr.shape[0])

            # Invert the top and right axes
            ax_top.invert_yaxis()   

            # Set the tick locations and labels for the top and right axes
            # chose 5 evenly spaced ticks
            x_values_espaced = np.linspace(0, x_dim, n_twin_labels, dtype=int)
            x_values_espaced_minor = np.linspace(0, x_dim, n_twin_labels*5, dtype=int)
            
            ax.set_xticks(x_values_espaced)
            ax.set_xticks(x_values_espaced_minor, minor=True)

            ax.set_yticks(x_values_espaced)
            ax.set_yticks(x_values_espaced_minor, minor=True)

            ax.set_xlim(x_corr_limits)
            ax.set_ylim(x_corr_limits)
            ax.invert_yaxis()



            y_values_espaced = np.linspace(0, y_dim, n_twin_labels, dtype=int)
            y_values_espaced_minor = np.linspace(0, y_dim, n_twin_labels*5, dtype=int)
            
            ax_top.set_xticks(y_values_espaced)
            ax_top.set_xticks(y_values_espaced_minor, minor=True)

            ax_right.set_yticks(y_values_espaced[::-1])
            ax_right.set_yticks(y_values_espaced_minor[::-1], minor=True)

            ax_top.set_xlim(y_corr_limits)
            ax_right.set_ylim(y_corr_limits[1], y_corr_limits[0])



            # shift the spines and set the labels
            ax_right.spines["right"].set_position(
                ("axes", 1 - (y_corr_limits[-1]-y_dim) / combined_corr.shape[0])
            )
            # write the labels in the correct place
            desired_ylabel_value = y_dim / 2
            transformed_ylabel_value_x, _ = ax_top.transAxes.inverted().transform(
                ax_top.transData.transform((desired_ylabel_value, 0))
            )
            _, transformed_ylabel_value_y = ax_right.transAxes.inverted().transform(
                ax_right.transData.transform((0, desired_ylabel_value))
            )

            ax_top.set_xlabel("Y Index", ha="center", va="bottom", x=transformed_ylabel_value_x)
            ax_right.set_ylabel("Y Index", ha="center", va="top", y=transformed_ylabel_value_y)

            # shift the x axes labels to the bottom to where y_right is equal to yaxis[0]
            ax.spines["bottom"].set_position(
                ("axes", (x_corr_limits[-1]-x_dim) / combined_corr.shape[0])
            )

            # write the labels in the correct place
            desired_x_value = x_dim / 2
            transformed_x_value_x, _ = ax.transAxes.inverted().transform(
                ax.transData.transform((desired_x_value, 0))
            )
            _ , transformed_x_value_y = ax.transAxes.inverted().transform(
                ax.transData.transform((0, desired_x_value))
            )

            # Set the x-axis label with the transformed x position
            ax.set_xlabel("X Index", ha="center", va="top", x=transformed_x_value_x)
            ax.set_ylabel("X Index", ha="center", va="bottom", y= transformed_x_value_y)

        # remove the spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
        ax_top.spines["bottom"].set_visible(False)
        ax_top.spines["left"].set_visible(False)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)
        ax_right.spines["bottom"].set_visible(False)
        ax_right.spines["left"].set_visible(False) 
        return fig, (ax, ax_right, ax_top)

    def plot_correlations(
        self,
        axis: int,
        cmap: str = "hot",
        n_per_row: int = 3,
        figsize: tuple = (6, 6),
        aspect="auto",
        apply_variable_alpha: bool = False,
        use_position_grid: bool = True,
        include_colorbars: bool = False,
        absolute: bool = True,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot the correlation matrices along the specified axis for each frequency index.

        Args:
            axis (int): The axis along which to calculate the correlation (0 for x, 1 for y).
            cmap (str, optional): The colormap to use for the plots. Defaults to "hot".
            n_per_row (int, optional): The number of plots per row. Defaults to 3.
            n_per_col (int, optional): The number of plots per column. Defaults to 3.
            figsize (tuple, optional): The figure size. Defaults to (6, 6).
            aspect (str, optional): The aspect ratio of the plots. Defaults to "auto".
            apply_variable_alpha (bool, optional): Whether to apply variable alpha based on the mean matrix. Defaults to False.
            use_position_grid (bool, optional): Whether to use the position grid for axis labels. Defaults to True.
            include_colorbars (bool, optional): Whether to include colorbars for each plot. Defaults to False.
            **kwargs: Additional keyword arguments to pass to imshow.

        Returns:
            Tuple[plt.Figure, np.ndarray]: The figure containing the plots and the array of axes.
        """
        n_plots = len(self.frequency_indices)
        n_rows = (n_plots - 1) // n_per_row + 1
        n_cols = n_per_row if n_plots > n_per_row else n_plots

        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
        if isinstance(axs, plt.Axes):
            axs = np.array([[axs]])
        if np.ndim(axs) == 1:
            axs = axs[np.newaxis, :]
        axs_flat = axs.flatten()

        for i, freq_index in enumerate(self.frequency_indices):
            ax = axs_flat[i]
            self.plot_correlation_at_index(
                index=freq_index,
                axis=axis,
                ax=ax,
                cmap=cmap,
                apply_variable_alpha=apply_variable_alpha,
                aspect=aspect,
                use_position_grid=use_position_grid,
                absolute=absolute,
                **kwargs,
            )
            ax.set_title(f"[{freq_index}]")
            if not include_colorbars:
                ax.images[-1].colorbar.remove()

        # Remove labels and ticks from inner plots
        for axx in axs[:-1].flatten():
            axx.set_xlabel("")
            axx.set_xticks([])
        for axx in axs[:, 1:].flatten():
            axx.set_ylabel("")
            axx.set_yticks([])

        # Remove unused subplots
        for i in range(n_plots, len(axs_flat)):
            axs_flat[i].axis("off")

        return fig, axs

    def plot_correlation_at_index(
        self,
        index=None,
        axis=0,
        ax=None,
        cmap: str = "hot",
        apply_variable_alpha: bool = True,
        aspect="auto",
        use_position_grid: bool = False,
        absolute: bool = True,
        **kwargs,
    ):
        """
        Plot the correlation matrix at the specified index along the given axis.

        Args:
            index (int, optional): The frequency index. Defaults to None (first index).
            axis (int): The axis along which to calculate the correlation (0 for x, 1 for y).
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None (creates a new figure).
            cmap (str, optional): The colormap to use for the plot. Defaults to "hot".
            apply_variable_alpha (bool, optional): Whether to apply variable alpha based on the mean matrix. Defaults to True.
            aspect (str, optional): The aspect ratio of the plot. Defaults to "auto".
            use_position_grid (bool, optional): Whether to use the position grid for axis labels. Defaults to False.
            **kwargs: Additional keyword arguments to pass to imshow.
        """
        if index is None:
            index = 0
        else:
            index = self.frequency_indices_order_dict.get(index, None)
            if index is None:
                raise ValueError("The frequency index is not in the list of frequency indices.")

        if axis == 0:
            correlation_matrix = self.x_correlation_matrix
            mean_matrix = self.x_mean_matrix
            axis_label = "x"
        elif axis == 1:
            correlation_matrix = self.y_correlation_matrix
            mean_matrix = self.y_mean_matrix
            axis_label = "y"
        else:
            raise ValueError("Axis must be 0 or 1.")

        if correlation_matrix is None:
            correlation_matrix = self.get_correlation_matrix(axis=axis)
        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)

        if apply_variable_alpha:
            if mean_matrix is None:
                mean_matrix = self.get_mean_matrix(axis=axis)
            mean_m = mean_matrix[..., index]
            if absolute:
                mean_m = np.abs(mean_m)
            alpha = (mean_m - mean_m.min()) / (mean_m.max() - mean_m.min())
        else:
            alpha = 1

        if use_position_grid:
            if self.position_grid is None:
                raise ValueError("Position grid is not available.")
            axis_values = getattr(self.position_grid, axis_label)
            extent = [axis_values.min(), axis_values.max(), axis_values.min(), axis_values.max()]
        else:
            extent = [0, self.data.shape[axis], 0, self.data.shape[axis]]

        if absolute:
            correlation_matrix = np.abs(correlation_matrix[..., index])
        else: 
            correlation_matrix = correlation_matrix[..., index]

        im = ax.imshow(
            correlation_matrix,
            extent=extent,
            aspect=aspect,
            cmap=cmap,
            alpha=alpha,
            **kwargs,
        )
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Correlation along {axis_label.upper()} axis")

        if use_position_grid:
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")
            ax.yaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")
            ax.set_xlabel(f"{axis_label} [mm]")
            ax.set_ylabel(f"{axis_label} [mm]")
        else:
            ax.set_xlabel(f"{axis_label}_index")
            ax.set_ylabel(f"{axis_label}_index")

    def plot_x_correlations(
        self,
        cmap: str = "hot",
        n_per_row: int = 3,
        figsize: tuple = (6, 6),
        aspect="auto",
        apply_variable_alpha: bool = False,
        use_position_grid: bool = True,
        include_colorbars: bool = False,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot the correlation matrices along the x-axis for each frequency index.

        Args:
            cmap (str, optional): The colormap to use for the plots. Defaults to "hot".
            n_per_row (int, optional): The number of plots per row. Defaults to 3.
            n_per_col (int, optional): The number of plots per column. Defaults to 3.
            figsize (tuple, optional): The figure size. Defaults to (10, 5).
            aspect (str, optional): The aspect ratio of the plots. Defaults to "auto".
            apply_variable_alpha (bool, optional): Whether to apply variable alpha based on the mean matrix. Defaults to True.
            **kwargs: Additional keyword arguments to pass to imshow.

        Returns:
            matplotlib.figure.Figure: The figure containing the plots.
        """
        return self.plot_correlations(
            axis=0,
            cmap=cmap,
            n_per_row=n_per_row,
            figsize=figsize,
            aspect=aspect,
            apply_variable_alpha=apply_variable_alpha,
            use_position_grid=use_position_grid,
            include_colorbars=include_colorbars,
            **kwargs,
        )

    def plot_y_correlations(
        self,
        cmap: str = "hot",
        n_per_row: int = 3,
        figsize: tuple = (6, 6),
        aspect="auto",
        include_colorbars: bool = False,
        apply_variable_alpha: bool = False,
        use_position_grid: bool = True,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot the correlation matrices along the y-axis for each frequency index.

        Args:
            cmap (str, optional): The colormap to use for the plots. Defaults to "hot".
            n_per_row (int, optional): The number of plots per row. Defaults to 3.
            n_per_col (int, optional): The number of plots per column. Defaults to 3.
            figsize (tuple, optional): The figure size. Defaults to (10, 5).
            aspect (str, optional): The aspect ratio of the plots. Defaults to "auto".
            apply_variable_alpha (bool, optional): Whether to apply variable alpha based on the mean matrix. Defaults to True.
            **kwargs: Additional keyword arguments to pass to imshow.

        Returns:
            matplotlib.figure.Figure: The figure containing the plots.
        """
        return self.plot_correlations(
            axis=1,
            cmap=cmap,
            n_per_row=n_per_row,
            figsize=figsize,
            aspect=aspect,
            apply_variable_alpha=apply_variable_alpha,
            use_position_grid=use_position_grid,
            include_colorbars=include_colorbars,
            **kwargs,
        )

    def plot_x_correlation_at_index(
        self,
        index=None,
        ax=None,
        cmap: str = "hot",
        apply_variable_alpha: bool = True,
        aspect="auto",
        use_position_grid: bool = False,
        **kwargs,
    ):
        return self.plot_correlation_at_index(
            index=index,
            axis=0,
            ax=ax,
            cmap=cmap,
            apply_variable_alpha=apply_variable_alpha,
            aspect=aspect,
            use_position_grid=use_position_grid,
            **kwargs,
        )

    def plot_y_correlation_at_index(
        self,
        index=None,
        ax=None,
        cmap: str = "hot",
        apply_variable_alpha: bool = True,
        aspect="auto",
        use_position_grid=False,
        **kwargs,
    ):
        return self.plot_correlation_at_index(
            index=index,
            axis=1,
            ax=ax,
            cmap=cmap,
            apply_variable_alpha=apply_variable_alpha,
            aspect=aspect,
            use_position_grid=use_position_grid,
            **kwargs,
        )

    def inspect_correlation_at_index(
        self,
        index=None,
        apply_variable_alpha: bool = True,
        units: str = "dBm",
        heatmap_title: str = "Scan Heatmap",
        cmap_heatmap: str = "jet",
        cmap_correlation: str = "hot",
        figsize: tuple = (10, 5),
        aspect="auto",
    ):

        fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        # Plot the heatmap
        q = axs[0, 0].imshow(self.data, aspect=aspect, cmap=cmap_heatmap)
        plt.colorbar(q, ax=axs[0, 0], label=f"[{units}]")
        axs[0, 0].set_xlabel("y_index")
        axs[0, 0].set_ylabel("x_index")
        axs[0, 0].set_title(heatmap_title, fontsize=16)

        # Use the new methods to plot the correlation matrices
        self.plot_x_correlation_at_index(
            index=index,
            ax=axs[1, 0],
            cmap=cmap_correlation,
            apply_variable_alpha=apply_variable_alpha,
            aspect=aspect,
        )
        self.plot_y_correlation_at_index(
            index=index,
            ax=axs[1, 1],
            cmap=cmap_correlation,
            apply_variable_alpha=apply_variable_alpha,
            aspect=aspect,
        )

        axs[1, 0].set_title(f"Correlation along X axis - [{index}]")
        axs[1, 1].set_title(f"Correlation along Y axis - [{index}]")

        return fig
