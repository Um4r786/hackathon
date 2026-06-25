from flask import Flask, render_template, request
import similarity_service

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("matintel.html", predictions=None, similar_materials=None, track=None)


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

    # Detect track from which properties the user supplied
    CHEMISTRY_PROPS = {"C_Wt_%", "Mn_Wt_%", "P_Wt_%", "S_Wt_%",
                       "Si_Wt_%", "Ni_Wt_%", "Cr_Wt_%", "Mo_Wt_%", "Ti_Wt_%"}
    PHYSICS_PROPS   = {"Density_(g/cm3)", "Energy_Above_Hull_(eV)",
                       "Bulk_Modulus_(GPa)", "Shear_Modulus_(GPa)"}

    supplied = {row["property"] for row in inputs}
    if supplied & CHEMISTRY_PROPS:
        track = "chemistry"
    elif supplied & PHYSICS_PROPS:
        track = "physics"
    else:
        track = "chemistry"   # default

    # Predicted properties (multi-select)
    predict_properties = request.form.getlist("predict_properties")

    # Only keep predictions that belong to the active track
    CHEMISTRY_OUTPUTS = {"Ultimate_Tensile_Strength_(MPa)", "Yield_Strength_(MPa)",
                         "Elongation_(%)", "Hardness_(HB)"}
    PHYSICS_OUTPUTS   = {"Density_(g/cm3)", "Energy_Above_Hull_(eV)",
                         "Bulk_Modulus_(GPa)", "Shear_Modulus_(GPa)"}
    allowed_outputs = CHEMISTRY_OUTPUTS if track == "chemistry" else PHYSICS_OUTPUTS

    # ---------- PLACEHOLDER MODEL LOGIC ----------
    # Replace the block below with a call to model_service.predict(user_inputs)
    # where user_inputs is a dict of {property: value} from inputs.

    predictions = []
    for prop in predict_properties:
        if prop in allowed_outputs:
            predictions.append({
                "property": prop,
                "value": 123.45,          # replace with model output
                "confidence": 0.92        # 0–1, will render as %
            })

    user_inputs = {row["property"]: row["value"] for row in inputs}

    similar_materials = similarity_service.get_top3_similar(
        material_id=material,
        input_properties=user_inputs,
    )
    print(similar_materials)
    # ---------------------------------------------

    return render_template(
        "matintel.html",
        predictions=predictions,
        similar_materials=similar_materials,
        material=material,
        inputs=inputs,
        track=track,
    )


if __name__ == "__main__":
    app.run(debug=True)