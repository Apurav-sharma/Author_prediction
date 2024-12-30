from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomExcep
import sys

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if(request.method == 'GET'):
        return render_template('pred_home.html')
    else:
        try:
            interlinear_spacing = float(request.form.get('interlinear_spacing'))
            modular_ratio = float(request.form.get('modular_ratio'))
            modular_ratio_by_interlinear_spacing = modular_ratio / interlinear_spacing
            data = CustomData(
                intercolumnar_distance = float(request.form.get('intercolumnar_distance')),
                upper_margin = float(request.form.get('upper_margin')),
                lower_margin = float(request.form.get('lower_margin')),
                exploitation = float(request.form.get('exploitation')),
                row_number = float(request.form.get('row_number')),
                modular_ratio = modular_ratio,
                interlinear_spacing = interlinear_spacing,
                weight = float(request.form.get('weight')),
                peak_number = float(request.form.get('peak_number')),
                modular_ratio_by_interlinear_spacing = modular_ratio_by_interlinear_spacing
            )

            df = data.get_data_frame()

            pipeline = PredictPipeline()
            prediction = pipeline.predict(df)

            return render_template('pred_home.html', prediction = prediction)
        except Exception as e:
            raise CustomExcep(e, sys)
    
if __name__ == '__main__':
    app.run()