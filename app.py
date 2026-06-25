from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("matintel.html", predictions=None, similar_materials=None)


@app.route("/predict", methods=["POST"])
def predict():
    material = request.form.get("material", "").strip()

    # Properties and values come as parallel lists
    properties = request.form.getlist("properties")
    values = request.form.getlist("values")

    # Filter out empty rows
    inputs = []
    for p, v in zip(properties, values):
        if p and v:
            try:
                inputs.append({"property": p, "value": float(v)})
            except ValueError:
                continue

    # Predicted properties (multi-select)
    predict_properties = request.form.getlist("predict_properties")

    # ---------- PLACEHOLDER MODEL LOGIC ----------
    # Here you would call your TensorFlow/Keras model with:
    #   - material
    #   - inputs (list of {property, value})
    #   - predict_properties
    #
    # For now, we mock some outputs.

    predictions = []
    for prop in predict_properties:
        predictions.append({
            "property": prop,
            "value": 123.45,          # replace with model output
            "confidence": 0.92        # 0–1, will render as %
        })

    similar_materials = [
        {
            "name": "Aluminium 6061",
            "similarity": 0.87,
            "key_properties": {
                "density": "2.70 g/cm³",
                "tensile_strength": "310 MPa"
            }
        },
        {
            "name": "Aluminium 7075",
            "similarity": 0.83,
            "key_properties": {
                "density": "2.81 g/cm³",
                "tensile_strength": "572 MPa"
            }
        }
    ]
    # ---------------------------------------------

    return render_template(
        "matintel.html",
        predictions=predictions,
        similar_materials=similar_materials,
        material=material,
        inputs=inputs,
    )


if __name__ == "__main__":
    app.run(debug=True)
