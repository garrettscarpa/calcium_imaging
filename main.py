import matplotlib.pyplot as plt
import traceback
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from data_loader import DataLoader
from ui_setup import build_ui
from file_handler import filetype_changed, select_file_or_folder, populate_file_list
from file_handler import validate_and_get_file_list
from utils import parse_analysis_parameters
from data_loader import load_and_preprocess
from peak_analysis import load_or_detect_peaks
from plotting import setup_plot_and_interaction, plot_all_rois_from_files


class CalciumImagingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calcium Imaging Analysis")
        self.csv_path = None
        self.npy_folder_path = None
        self.figures = []  # Initialize the list of figures

        # Build UI layout and retrieve all widgets
        layout, ui = build_ui(self)
        self.setLayout(layout)

        # Assign widgets to instance attributes
        self.file_list_widget = ui['file_list_widget']
        self.filetype_combo = ui['filetype_combo']
        self.file_button = ui['file_button']
        self.file_label = ui['file_label']
        self.npy_filename_input = ui['npy_filename_input']
        self.fs_input = ui['fs_input']
        self.truncate_seconds_input = ui['truncate_seconds_input']
        self.prominence_input = ui['prominence_input']
        self.min_height_input = ui['min_height_input']
        self.min_plateau_input = ui['min_plateau_input']
        self.poly_order_input = ui['poly_order_input']
        self.smoothing_window_input = ui['smoothing_window_input']
        self.display_window_input = ui['display_window_input']
        self.yax_input = ui['yax_input']
        self.transpose_checkbox = ui['transpose_checkbox']
        self.pre_range_input = ui['pre_range_input']
        self.post_range_input = ui['post_range_input']
        self.run_button = ui['run_button']
        self.plot_all_button = ui['plot_all_button']
        self.cell_filter_checkbox = ui['cell_filter_checkbox']

        # Connect signals to methods and external handlers
        self.filetype_combo.currentIndexChanged.connect(lambda: filetype_changed(self))
        self.file_button.clicked.connect(lambda: select_file_or_folder(self))
        self.npy_filename_input.textChanged.connect(lambda: populate_file_list(self))
        self.run_button.clicked.connect(self.run_analysis)
        self.plot_all_button.clicked.connect(self.plot_all_rois)

    def plot_all_rois(self):
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No selection", "Please select one .npy recording from the list.")
            return
    
        selected_file = selected_items[0].text()
        if not selected_file.lower().endswith('.npy'):
            QMessageBox.warning(self, "Invalid file type", "Only .npy files can be plotted with 'Plot All ROIs'.")
            return
    
        try:
            fs = float(self.fs_input.text())  # Sampling rate in Hz (samples per second)
            truncate_seconds = float(self.truncate_seconds_input.text())
            truncate_samples = int(truncate_seconds * fs)  # Number of samples to truncate
            transpose = self.transpose_checkbox.isChecked()
            smoothing_window_length = int(self.smoothing_window_input.text())
            poly_order = int(self.poly_order_input.text())
            y_ax_range = tuple(map(float, self.yax_input.text().split(',')))
            display_window = float(self.display_window_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid numeric values.")
            return
    
        # Load & preprocess with iscell filtering
        data, smoothed_dff_df = load_and_preprocess(
            selected_file,
            "NPY",
            transpose,
            truncate_samples,
            smoothing_window_length,
            poly_order,
            include_only_cells=self.cell_filter_checkbox.isChecked()
        )
        print(smoothed_dff_df)
        # Create time vector in seconds (X-axis in seconds)
        time = smoothed_dff_df.columns / fs  # Convert sample indices to time (seconds)
    
        # Plot
        fig, ax = plt.subplots()
        smoothed_dff_df.T.plot(ax=ax)
    
        # Auto scale y-axis based on data
        y_min = smoothed_dff_df.min().min()
        y_max = smoothed_dff_df.max().max()
        padding = 0.05 * (y_max - y_min)  # 5% padding
        ax.set_ylim(y_min - padding, y_max + padding)
    
        # Set the x-axis to be time in seconds
        ax.set_xticks(ax.get_xticks())  # Get current tick positions
        ax.set_xticklabels([f"{x/fs:.2f}" for x in ax.get_xticks()])  # Convert to time in seconds
    
        # Label x-axis as 'Time (s)'
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ΔF/F')
    
        plt.show()
    
    def run_analysis(self):
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No selection", "Please select one .npy recording from the list.")
            return
    
        selected_file = selected_items[0].text()
        if not selected_file.lower().endswith('.npy'):
            QMessageBox.warning(self, "Invalid file type", "Only .npy files can be plotted with 'Plot All ROIs'.")
            return
    
        try:
            # Retrieve and validate inputs
            fs = float(self.fs_input.text())  # Sampling rate in Hz (samples per second)
            truncate_seconds = float(self.truncate_seconds_input.text())
            truncate_samples = int(truncate_seconds * fs)  # Number of samples to truncate
            transpose = self.transpose_checkbox.isChecked()
            smoothing_window_length = int(self.smoothing_window_input.text())
            poly_order = int(self.poly_order_input.text())
            y_ax_range = tuple(map(float, self.yax_input.text().split(',')))
            display_window = float(self.display_window_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid numeric values.")
            return
    
        # Now handle the parameters with more validation
        try:
            # Check if the fields are empty, and set default values if necessary
            prominence = float(self.prominence_input.text()) if self.prominence_input.text() else 0.1
            min_height = float(self.min_height_input.text()) if self.min_height_input.text() else 0.0
            min_plateau_samples = int(self.min_plateau_input.text()) if self.min_plateau_input.text() else 3
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid numeric values for prominence, min height, and plateau samples.")
            return
    
        # Create the params dictionary
        params = {
            'fs': fs,
            'prominence': prominence,
            'min_height': min_height,
            'min_plateau_samples': min_plateau_samples,
            'display_window': display_window,
            'y_ax_range': y_ax_range
        }
    
        # Load & preprocess with iscell filtering
        data, smoothed_dff_df = load_and_preprocess(
            selected_file,
            "NPY",
            transpose,
            truncate_samples,
            smoothing_window_length,
            poly_order,
            include_only_cells=self.cell_filter_checkbox.isChecked()
        )
    
        print(smoothed_dff_df)
    
        # Load peaks
        peak_results_df, export_dir, file_prefix = load_or_detect_peaks(
            self, selected_file, smoothed_dff_df, params['fs'], params['prominence'],
            params['min_height'], params['min_plateau_samples']
        )
        if peak_results_df is None:
            return
    
        # Set up the plot and interaction
        fig = setup_plot_and_interaction(
            self, smoothed_dff_df, peak_results_df, params['fs'],
            params['display_window'], params['y_ax_range'],
            export_dir, file_prefix, DataLoader(data, params['fs'])
        )
    
        try:
            fig.canvas.manager.window.activateWindow()
            fig.canvas.manager.window.raise_()
        except Exception as e:
            print(f"Could not raise figure window: {e}")
    
        # Append the figure to the list
        self.figures.append(fig)
    
        plt.show()



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = CalciumImagingApp()
    window.resize(400, 600)
    window.show()
    sys.exit(app.exec_())
