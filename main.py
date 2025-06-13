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

        # Connect signals to methods and external handlers
        self.filetype_combo.currentIndexChanged.connect(lambda: filetype_changed(self))
        self.file_button.clicked.connect(lambda: select_file_or_folder(self))
        self.npy_filename_input.textChanged.connect(lambda: populate_file_list(self))
        self.run_button.clicked.connect(self.run_analysis)
        self.plot_all_button.clicked.connect(self.plot_all_rois)

    def plot_all_rois(self):    
        # Ensure one file is selected
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No selection", "Please select one recording from the list.")
            return
    
        selected_file = selected_items[0].text()  # Only the first selected file
    
        # Parse numeric inputs
        try:
            fs = float(self.fs_input.text())
            truncate_seconds = float(self.truncate_seconds_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid numeric values for sampling frequency and truncation.")
            return
    
        transpose = self.transpose_checkbox.isChecked()
    
        # Call the function from plotting.py
        plot_all_rois_from_files(
            file_list=[selected_file],  # Use only the selected file
            fs=fs,
            truncate_seconds=truncate_seconds,
            transpose=transpose,
            ca_marker='*',  # Customize if needed
            display_window=float(self.display_window_input.text()),
            y_ax_range=tuple(map(float, self.yax_input.text().split(',')))  # expects "ymin,ymax"
        )


    def run_analysis(self):
        selected_type = self.filetype_combo.currentText()
    
        file_list = validate_and_get_file_list(self, selected_type)
        if file_list is None:
            return
    
        params = parse_analysis_parameters(self)
        if params is None:
            QMessageBox.warning(self, "Error", "Please enter valid numeric values for all parameters.")
            return
    
        transpose = self.transpose_checkbox.isChecked()
        figures = []
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No selection", "Please select at least one recording from the list.")
            return
    
        for item in selected_items:
            file_path = item.text()
            try:
                data, smoothed_dff_df = load_and_preprocess(
                    file_path, selected_type, transpose,
                    params['truncate_samples'], params['smoothing_window_length'], params['polynomial_order']
                )
    
                loader = DataLoader(data, params['fs'])
    
                peak_results_df, export_dir, file_prefix = load_or_detect_peaks(
                    self, file_path, smoothed_dff_df, params['fs'], params['prominence'],
                    params['min_height'], params['min_plateau_samples']
                )
                if peak_results_df is None:
                    return
    
                fig = setup_plot_and_interaction(
                    self, smoothed_dff_df, peak_results_df, params['fs'],
                    params['display_window'], params['y_ax_range'],
                    export_dir, file_prefix, loader
                )
    
                try:
                    fig.canvas.manager.window.activateWindow()
                    fig.canvas.manager.window.raise_()
                except Exception as e:
                    print(f"Could not raise figure window: {e}")
    
                figures.append(fig)
    
            except Exception as e:
                import traceback
                traceback_str = traceback.format_exc()
                print(f"[ERROR] Failed to process file: {file_path}\n{traceback_str}")
                QMessageBox.critical(self, "Error", f"Error processing {file_path}:\n\n{str(e)}")

    
        plt.show()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = CalciumImagingApp()
    window.resize(400, 600)
    window.show()
    sys.exit(app.exec_())
