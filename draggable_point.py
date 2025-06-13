import numpy as np

class DraggablePoint:
    def __init__(self, ax, x, y, color, label, time_axis, trace, fs, smoothed_dff_df, current_roi, plotter):
        self.ax = ax
        self.x = x
        self.y = y
        self.color = color
        self.label = label
        self.time_axis = time_axis
        self.trace = trace
        self.plot, = ax.plot([x], [y], 'o', color=color, label=label, picker=True)
        self.cid = self.plot.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.dragging = False
        self.fs = fs
        self.smoothed_dff_df = smoothed_dff_df
        self.current_roi = current_roi
        self.plotter = plotter

    def on_pick(self, event):
        if event.artist != self.plot:
            return
        self.dragging = True
        self.plot.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.plot.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_motion(self, event):
        if self.dragging and event.xdata is not None:
            min_x = self.time_axis[0]
            max_x = self.time_axis[-1]
            self.x = np.clip(event.xdata, min_x, max_x)
            self.update_base_value()  # ✅ No longer raises error
            self.update_plot()

    def on_release(self, event):
        self.dragging = False
        self.update_data()

    def update_plot(self):
        self.plot.set_data([self.x], [self.y])  # <-- wrap in list
        self.ax.figure.canvas.draw()


    def update_base_value(self):
        # ✅ Now defined: dynamically calculate y using interpolation
        self.y = np.interp(self.x, self.time_axis, self.trace)

    def update_data(self):
        # Access required data via plotter
        peak_results_df = self.plotter.peak_results_df
        peaks_in_roi = self.plotter.peaks_in_roi
        peak_idx = self.plotter.peak_idx
        change_history = self.plotter.change_history

        peak_row = peaks_in_roi.iloc[peak_idx]
        peak_index = peak_row.name
        change_history.append(('move_base', peak_row.copy(), peak_results_df.copy()))

        if self.label == 'Left Base':
            peak_results_df.at[peak_index, 'left_bases'] = self.x
            peak_results_df.at[peak_index, 'base_value'] = self.y
        elif self.label == 'Right Base':
            peak_results_df.at[peak_index, 'right_bases'] = self.x

        left_base_time = peak_results_df.at[peak_index, 'left_bases']
        right_base_time = peak_results_df.at[peak_index, 'right_bases']
        left_idx = int(left_base_time * self.fs)
        right_idx = int(right_base_time * self.fs)

        trace = self.smoothed_dff_df.loc[self.current_roi].values
        time_axis = np.arange(len(trace)) / self.fs

        if left_idx >= right_idx or left_idx < 0 or right_idx >= len(trace):
            print("Invalid base window selected.")
            return

        window_trace = trace[left_idx:right_idx + 1]
        window_time = time_axis[left_idx:right_idx + 1]

        new_peak_idx = np.argmax(window_trace)
        new_peak_time = window_time[new_peak_idx]
        new_peak_value = window_trace[new_peak_idx]

        left_base_val = np.interp(left_base_time, time_axis, trace)
        right_base_val = np.interp(right_base_time, time_axis, trace)
        new_base_value = min(left_base_val, right_base_val)
        new_prominence = new_peak_value - new_base_value

        # Recalculate half-rise
        half_max = new_base_value + 0.5 * (new_peak_value - new_base_value)
        half_rise_time = np.nan
        for j in range(left_idx, left_idx + new_peak_idx):
            if trace[j] <= half_max <= trace[j + 1]:
                frac = (half_max - trace[j]) / (trace[j + 1] - trace[j])
                half_rise_time = (j + frac - left_idx) / self.fs
                break

        # Recalculate half-decay
        half_decay_time = np.nan
        for j in range(left_idx + new_peak_idx, right_idx):
            if trace[j] >= half_max >= trace[j + 1]:
                frac = (trace[j] - half_max) / (trace[j] - trace[j + 1])
                half_decay_time = (j + frac - (left_idx + new_peak_idx)) / self.fs
                break

        peak_results_df.at[peak_index, 'half_rise_time'] = half_rise_time
        peak_results_df.at[peak_index, 'half_decay_time'] = half_decay_time
        peak_results_df.at[peak_index, 'peak_time'] = new_peak_time
        peak_results_df.at[peak_index, 'peak_value'] = new_peak_value
        peak_results_df.at[peak_index, 'base_value'] = new_base_value
        peak_results_df.at[peak_index, 'prominences'] = new_prominence
        peak_results_df.at[peak_index, 'time_to_peak'] = peak_results_df.at[peak_index, 'peak_time'] - peak_results_df.at[peak_index, 'left_bases']

        auc_values = trace[left_idx:right_idx + 1]
        auc_values_abs = np.abs(auc_values - new_base_value)
        auc_value = np.trapz(auc_values_abs, dx=1 / self.fs)
        peak_results_df.at[peak_index, 'auc'] = auc_value

        self.plotter.update_plot()

