from flask import Flask, render_template, request

app = Flask(__name__)

# loads webpage
@app.route("/")
def index():
    return render_template('index.html')

# processes form input
@app.route("/search", methods=["GET"])
def process():
    material = request.args.get("material")
    properties_raw = request.args.get("properties")

    # split comma‑separated properties
    properties = [p.strip() for p in properties_raw.split(",")]

    # static dummy values
    dummy_rankings = {
        prop: {
            "ranking": i + 1,
            "confidence": f"{90 - i*5}%"   # 90%, 85%, 80%, ...
        }
        for i, prop in enumerate(properties)
    }

    return render_template(
        "results.html",
        material=material,
        rankings=dummy_rankings
    )

# explain page
@app.route("/explain/<property_name>")
def explain(property_name):
    return render_template("explain.html", property_name=property_name)

if __name__ == "__main__":
    app.run(debug=True)