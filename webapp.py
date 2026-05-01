from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


def wage(k, alpha):
    return (1 - alpha) * (k ** alpha)


def interest(k, alpha, delta):
    return alpha * (k ** (alpha - 1.0)) - delta


def saving_share(r, rho, theta, tau_k):
    if r <= 0:
        return 0.0
    base = (1 - tau_k) * r
    if base <= 0:
        return 0.0
    num = base ** ((1 - theta) / theta)
    den = (1 + rho) ** (1 / theta) + num
    return num / den


@app.get('/')
def landing():
    return render_template('landing.html')


@app.get('/demo')
def demo():
    return render_template('demo.html')


@app.get('/model')
def model():
    return render_template('model.html')


@app.post('/api/run-model')
def run_model():
    p = request.get_json(force=True)
    alpha = float(p.get('alpha', 0.33))
    delta = float(p.get('delta', 0.05))
    rho = float(p.get('rho', 0.04))
    theta = float(p.get('theta', 2.0))
    tau_k = float(p.get('tau_k', 0.10))
    k = float(p.get('k', 0.8))

    k = max(k, 1e-6)
    w = wage(k, alpha)
    r = interest(k, alpha, delta)
    s = saving_share(r, rho, theta, tau_k)

    return jsonify({
        'wage': round(w, 6),
        'interest': round(r, 6),
        'saving_share': round(s, 6),
        'status': 'Model run completed successfully.'
    })


if __name__ == '__main__':
    app.run(debug=True)
