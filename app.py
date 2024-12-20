from flask import Flask, render_template, request
import re
import main

app = Flask(__name__)

def format_bold_text(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

@app.route("/", methods=["GET", "POST"])
def home():
    response_message = None
    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        symptoms = request.form.get("symptoms")

        user_input = f"{name}. {age}. {symptoms}"

        try:
            response_message = main.process_input(user_input)
            response_message = format_bold_text(response_message)
            response_message = response_message.replace("\n", "<br>")
        except Exception as e:
            response_message = f"Erro ao processar a solicitação: {str(e)}"

    return render_template("index.html", response_message=response_message)

if __name__ == "__main__":
    app.run(debug=True)
