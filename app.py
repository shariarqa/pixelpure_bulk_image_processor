from flask import Flask, render_template, request, redirect, send_file
import os
import shutil
from your_script import ImageRenamerThread  # Import your processing logic

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/site/wwwroot/uploads/'
app.config['PROCESSED_FOLDER'] = '/home/site/wwwroot/processed/'


def clean_directory(directory):
    """Helper function to delete all files in a directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('file')
        stock_site = request.form.get('stock_site')

        # Save uploaded files
        for file in files:
            if file:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

        # Process the uploaded images
        renamer = ImageRenamerThread(app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'], stock_site)
        renamer.run()

        # Generate the CSV
        csv_file = renamer.save_csv([], app.config['PROCESSED_FOLDER'], len(files))

        # Clean up: Delete the uploaded and processed images
        clean_directory(app.config['UPLOAD_FOLDER'])
        clean_directory(app.config['PROCESSED_FOLDER'])

        # Provide the CSV file for download
        return send_file(csv_file, as_attachment=True)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
