from flask import Flask, request
app = Flask(__name__)

@app.route("/test_candidate.py")
def hello():
    a = request.args.get('a', type=int)
    b = request.args.get('b', type=int)
    c = request.args.get('c', type=int)
    return str(a * b * c)

if __name__ == "__main__":
    app.run()