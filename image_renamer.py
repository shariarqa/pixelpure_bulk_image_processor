import sys
import os
import re
import csv
import random
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit, \
    QMessageBox, QProgressBar, QGroupBox, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image

# Adobe Stock Categories
ADOBE_STOCK_CATEGORIES = {
    "Animals": 1, "Buildings and Architecture": 2, "Business": 3, "Drinks": 4,
    "The Environment": 5, "States of Mind": 6, "Food": 7, "Graphic Resources": 8,
    "Hobbies and Leisure": 9, "Industry": 10, "Landscape": 11, "Lifestyle": 12,
    "People": 13, "Plants and Flowers": 14, "Culture and Religion": 15,
    "Science": 16, "Social Issues": 17, "Sports": 18, "Technology": 19,
    "Transport": 20, "Travel": 21
}

# Shutterstock Categories
SHUTTERSTOCK_CATEGORIES = [
    "Abstract", "Animals/Wildlife", "Architecture", "Arts and Entertainment",
    "Business/Finance", "Education", "Fashion", "Food/Drink", "Health/Medical",
    "Holidays/Celebrations", "Industry/Crafts", "Nature", "People", "Religion",
    "Science/Technology", "Sports/Recreation", "Transportation", "Travel/Destinations",
    "Vintage", "Vectors/Illustrations"
]

# Initialize the BART model for paraphrasing
paraphraser = pipeline("text2text-generation", model="facebook/bart-large-cnn")

class ImageRenamerThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(self, input_folder, output_folder, stock_site):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.stock_site = stock_site
        self._stop_flag = False

    def run(self):
        existing_filenames = set(os.listdir(self.output_folder))
        total_files = len([filename for filename in os.listdir(self.input_folder) if
                           filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])

        csv_data = []

        # Load BLIP model and processor
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        incremental_number = 1  # Start incremental number
        date_str = datetime.now().strftime("%Y%m%d")  # Get the current date in YYYYMMDD format

        for i, filename in enumerate(os.listdir(self.input_folder)):
            if self._stop_flag:
                self.stopped.emit()
                return

            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(self.input_folder, filename)
                image = Image.open(file_path).convert("RGB")

                # Generate caption using BLIP
                inputs = processor(images=image, return_tensors="pt")
                outputs = model.generate(**inputs)
                blip_caption = processor.decode(outputs[0], skip_special_tokens=True)

                # Sanitize title
                title = self.sanitize_title(blip_caption)

                # Paraphrase the title for better readability
                final_title = self.paraphrase_text(title)

                # Generate keywords based on the title
                keywords = self.generate_keywords(final_title)

                # Remove institution names and website addresses from keywords
                keywords = self.remove_institution_names_and_websites_from_keywords(keywords)

                # Automatically select category based on keywords
                selected_category = self.select_category_based_on_keywords(keywords)

                # Generate a unique filename with numeric date and incremental numbering
                original_extension = os.path.splitext(filename)[1]
                new_filename = self.get_incremental_filename(date_str, incremental_number)
                incremental_number += 1

                new_file_path = os.path.join(self.output_folder, f"{new_filename}{original_extension}")

                # Rename the file by moving it to the output folder
                os.rename(file_path, new_file_path)
                existing_filenames.add(new_filename + original_extension)

                # Prepare CSV data based on stock site
                if self.stock_site == "Adobe Stock":
                    row = {
                        "Filename": f"{new_filename}{original_extension}",
                        "Title": final_title,
                        "Keywords": ", ".join(keywords),
                        "Category": selected_category,
                        "Releases": ""
                    }
                elif self.stock_site == "Shutterstock":
                    categories = self.select_shutterstock_categories(keywords)
                    row = {
                        "Filename": f"{new_filename}{original_extension}",
                        "Description": final_title,
                        "Keywords": ", ".join(keywords),
                        "Categories": ", ".join(categories),
                        "Editorial": "no",
                        "Mature Content": "no",
                        "Illustration": "no"
                    }

                csv_data.append(row)

                # Update progress bar
                progress_percentage = int((i + 1) / total_files * 100)
                self.progress.emit(progress_percentage)

        # Save CSV
        self.save_csv(csv_data, self.output_folder, total_files)

        self.finished.emit()

    def stop(self):
        self._stop_flag = True

    def sanitize_title(self, title):
        """Sanitize the title by removing unwanted elements."""
        # Remove "¬"
        title = title.replace("¬", "")

        # Remove any web addresses and phrases that include them
        title = re.sub(r'(https?://|www\.)\S+', '', title)

        # Remove any institution names
        institution_names = ["CNN", "BBC", "Harvard", "MIT"]  # Add more as needed
        for name in institution_names:
            title = re.sub(r'\b' + re.escape(name) + r'\b', '', title, flags=re.IGNORECASE)

        # Remove special characters
        title = re.sub(r'[^\w\s]', '', title)

        # Remove promotional phrases and requests to visit any place or website
        title = re.sub(
            r'(for more,?\s*go to\s*|visit\s+\S+|CNN\.com.*?gallery|submit.*?shots.*?week|Please submit.*?shots|visit.*?next\s+Wednesday).*?$',
            '',
            title,
            flags=re.IGNORECASE
        )

        # Remove extra spaces after cleaning
        title = re.sub(r'\s+', ' ', title).strip()

        # Ensure title is less than 200 characters
        if len(title) > 200:
            title = title[:200].rsplit(' ', 1)[0]

        return title.strip()

    def paraphrase_text(self, text):
        """Paraphrase the title to ensure readability and uniqueness."""
        paraphrased_output = paraphraser(text, max_length=200, min_length=50, do_sample=False)
        paraphrased_text = paraphrased_output[0]['generated_text']

        # Remove repeated words
        words = paraphrased_text.split()
        seen = set()
        unique_words = [word for word in words if not (word in seen or seen.add(word))]

        # Join words back into a sentence
        final_title = ' '.join(unique_words)

        # Ensure the final title is a complete sentence
        if not final_title.endswith('.'):
            final_title += '.'

        return final_title.strip()

    def generate_keywords(self, title):
        """Generates keywords based on the title."""
        ENGLISH_STOP_WORDS = set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for',
            'if', 'in', 'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or',
            'such', 'that', 'the', 'their', 'then', 'there', 'these', 'they',
            'this', 'to', 'was', 'will', 'with'
        ])

        words = set(re.findall(r'\b\w+\b', title))
        words = words.difference(ENGLISH_STOP_WORDS)
        words = {word for word in words if len(word) > 1 and word.isalpha()}

        if len(words) > 50:
            keywords = random.sample(words, 50)
        else:
            keywords = list(words)

        return keywords

    def remove_institution_names_and_websites_from_keywords(self, keywords):
        """Removes any institution names and website addresses from the keywords."""
        institution_names = {"CNN", "BBC", "Harvard", "MIT"}  # Add more as needed
        keywords = [keyword for keyword in keywords if keyword not in institution_names]

        # Remove any keywords that resemble a URL
        keywords = [keyword for keyword in keywords if not re.match(r'(https?://|www\.)\S+', keyword)]

        return keywords

    def select_category_based_on_keywords(self, keywords):
        """Selects the appropriate Adobe Stock category based on keywords."""
        for keyword in keywords:
            for category, category_num in ADOBE_STOCK_CATEGORIES.items():
                if keyword.lower() in category.lower():
                    return str(category_num)
        return "1"  # Default category if no match is found

    def select_shutterstock_categories(self, keywords):
        """Selects two appropriate Shutterstock categories based on keywords."""
        selected_categories = []
        for keyword in keywords:
            for category in SHUTTERSTOCK_CATEGORIES:
                if category.lower() in keyword.lower() and category not in selected_categories:
                    selected_categories.append(category)
                if len(selected_categories) == 2:
                    break
            if len(selected_categories) == 2:
                break

        if not selected_categories:
            selected_categories = ["Nature", "People"]  # Default categories if no match is found

        return selected_categories

    def get_incremental_filename(self, date_str, incremental_number):
        """Generates a filename with numeric date and incremental numbering for the selected stock site."""
        prefix = "adobe" if self.stock_site == "Adobe Stock" else "shutterstock"
        return f"{prefix}_{date_str}_{incremental_number}"

    def save_csv(self, data, output_folder, total_photos):
        """Saves the data into a CSV file for the selected stock site."""
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        stock_site = "AdobeStock" if self.stock_site == "Adobe Stock" else "Shutterstock"
        csv_filename = os.path.join(output_folder, f"{stock_site}_{total_photos}Photos_{now}.csv")

        if self.stock_site == "Adobe Stock":
            columns = ["Filename", "Title", "Keywords", "Category", "Releases"]
        elif self.stock_site == "Shutterstock":
            columns = ["Filename", "Description", "Keywords", "Categories", "Editorial", "Mature Content", "Illustration"]

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for row in data:
                filtered_row = {k: v for k, v in row.items() if k in columns}
                writer.writerow(filtered_row)

        return csv_filename

class ImageRenamerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PixelPure Stock Image Processor Version 1')
        self.setGeometry(100, 100, 600, 600)

        layout = QVBoxLayout()

        self.headline_label = QLabel('<a href="https://example.com">PixelPure Stock Image Processor Version 1</a>')
        self.headline_label.setAlignment(Qt.AlignCenter)
        self.headline_label.setOpenExternalLinks(True)
        self.headline_label.setStyleSheet("color: #00e6e6; font-size: 18pt; font-weight: bold;")
        layout.addWidget(self.headline_label)

        self.stock_site_label = QLabel('Select Stock Photo Site:')
        layout.addWidget(self.stock_site_label)

        self.stock_site_combo = QComboBox(self)
        self.stock_site_combo.addItems(["Adobe Stock", "Shutterstock"])
        layout.addWidget(self.stock_site_combo)

        self.input_label = QLabel('Select Input Folder:')
        layout.addWidget(self.input_label)

        self.input_button = QPushButton('Browse Input Folder')
        self.input_button.clicked.connect(self.select_input_folder)
        layout.addWidget(self.input_button)

        self.output_label = QLabel('Select Output Folder:')
        layout.addWidget(self.output_label)

        self.output_button = QPushButton('Browse Output Folder')
        self.output_button.clicked.connect(self.select_output_folder)
        layout.addWidget(self.output_button)

        self.before_label = QLabel('Text to add before the filename (optional):')
        layout.addWidget(self.before_label)

        self.before_input = QLineEdit(self)
        layout.addWidget(self.before_input)

        self.after_label = QLabel('Text to add after the filename (optional):')
        layout.addWidget(self.after_label)

        self.after_input = QLineEdit(self)
        layout.addWidget(self.after_input)

        self.progress_group = QGroupBox("Progress Bar")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #87CEEB; width: 20px; }")
        self.progress_bar.setFormat('%p%')
        self.progress_bar.setFixedHeight(30)
        progress_layout.addWidget(self.progress_bar)

        self.progress_group.setLayout(progress_layout)
        layout.addWidget(self.progress_group)

        self.run_button = QPushButton('Run')
        self.run_button.clicked.connect(self.run)
        self.run_button.setStyleSheet("background-color: grey; font-weight: bold;")
        layout.addWidget(self.run_button)

        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        layout.addWidget(self.stop_button)

        self.resume_button = QPushButton('Resume')
        self.resume_button.clicked.connect(self.resume)
        self.resume_button.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        layout.addWidget(self.resume_button)

        exit_button_layout = QHBoxLayout()
        self.exit_button = QPushButton('Exit')
        self.exit_button.clicked.connect(self.exit_application)
        self.exit_button.setStyleSheet("background-color: #333333; color: #ff0000; font-weight: bold;")
        exit_button_layout.addWidget(self.exit_button)

        exit_button_widget = QWidget()
        exit_button_widget.setLayout(exit_button_layout)
        layout.addWidget(exit_button_widget, alignment=Qt.AlignLeft)

        self.setLayout(layout)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Input Folder')
        if folder:
            self.input_label.setText(f'Selected Input Folder: {folder}')
            self.input_folder = folder

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if folder:
            self.output_label.setText(f'Selected Output Folder: {folder}')
            self.output_folder = folder

    def run(self):
        if not hasattr(self, 'input_folder') or not hasattr(self, 'output_folder'):
            QMessageBox.warning(self, 'Input/Output Missing', 'Please select both input and output folders.')
            return

        self.run_button.setStyleSheet("background-color: green; font-weight: bold;")
        self.stop_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.exit_button.setEnabled(False)

        # Disable before/after text inputs when running
        self.before_input.setEnabled(False)
        self.after_input.setEnabled(False)

        stock_site = self.stock_site_combo.currentText()

        self.thread = ImageRenamerThread(self.input_folder, self.output_folder, stock_site)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.process_finished)
        self.thread.stopped.connect(self.process_stopped)
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()

    def resume(self):
        if self.progress_bar.value() > 0 and self.progress_bar.value() < self.progress_bar.maximum():
            self.run()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def process_finished(self):
        self.run_button.setStyleSheet("background-color: grey; font-weight: bold;")
        self.stop_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.exit_button.setEnabled(True)

        # Re-enable before/after text inputs after processing
        self.before_input.setEnabled(True)
        self.after_input.setEnabled(True)

        QMessageBox.information(self, 'Success', 'Images processed successfully!')

    def process_stopped(self):
        self.run_button.setStyleSheet("background-color: grey; font-weight: bold;")
        self.resume_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.exit_button.setEnabled(True)

        # Re-enable before/after text inputs after processing
        self.before_input.setEnabled(True)
        self.after_input.setEnabled(True)

        QMessageBox.information(self, 'Stopped', 'Operation Stopped!')

    def exit_application(self):
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, 'Error', 'Process running, please stop before exiting.')
        else:
            self.close()

def main():
    app = QApplication(sys.argv)

    ex = ImageRenamerApp()
    ex.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
