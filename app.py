from flask import Flask, render_template, request, jsonify
import numpy as np


# app = Flask(__name__)


app = Flask(__name__, static_folder='static') # Important!



class EuropeanOptionPricing:
    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_simulation(self):
        # we have 2 columns: first with 0s the second column will store the payoff
        # we need the first column of 0s: payoff function is max(0,S-E) for call option
        option_data = np.zeros([self.iterations, 2])

        # dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for the S(t) stock price at T
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.T) * rand)

        # we need S-E because we have to calculate the max(S-E,0)
        option_data[:, 1] = stock_price - self.E

        # average for the Monte-Carlo simulation
        # max() returns the max(0,S-E) according to the formula
        # THIS IS THE AVERAGE VALUE !!!
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)

        # have to use the exp(-rT) discount factor
        return np.exp(-1.0 * self.rf * self.T) * average

    def put_option_simulation(self):
        # we have 2 columns: first with 0s the second column will store the payoff
        # we need the first column of 0s: payoff function is max(0,S-E) for call option
        option_data = np.zeros([self.iterations, 2])

        # dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for the S(t) stock price at T
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2)
                                       + self.sigma * np.sqrt(self.T) * rand)

        # we need S-E because we have to calculate the max(E-S,0)
        option_data[:, 1] = self.E - stock_price

        # average for the Monte-Carlo simulation
        # max() returns the max(0,S-E) according to the formula
        # THIS IS THE AVERAGE VALUE !!!
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)

        # have to use the exp(-rT) discount factor
        return np.exp(-1.0 * self.rf * self.T) * average


class AmericanOptionPricing:
    def __init__(self, S0, K, T, r, sigma, steps, simulations):
        self.S0 = S0
        self.K = K  # Strike price
        self.T = T
        self.r = r  # Risk-free rate
        self.sigma = sigma
        self.steps = steps
        self.simulations = simulations
        self.dt = T / steps

    def stock_price_simulation(self):
        # Initialize stock price array (simulations x steps+1)
        S = np.zeros((self.simulations, self.steps + 1))
        S[:, 0] = self.S0

        # Generate random normal numbers
        Z = np.random.standard_normal(size=(self.simulations, self.steps))

        # Simulate stock price paths
        for t in range(1, self.steps + 1):
            S[:, t] = S[:, t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt +
                                           self.sigma * np.sqrt(self.dt) * Z[:, t - 1])
        return S

    def american_call_option(self):
        # Simulate stock price paths
        S = self.stock_price_simulation()

        # Initialize payoff array
        payoff = np.zeros((self.simulations, self.steps + 1))

        # Compute payoff at expiration (last step)
        payoff[:, -1] = np.maximum(S[:, -1] - self.K, 0)

        # Work backwards through time
        for t in range(self.steps - 1, -1, -1):
            # Calculate immediate exercise value
            immediate_exercise = np.maximum(S[:, t] - self.K, 0)

            # Identify in-the-money paths
            itm = immediate_exercise > 0

            # Continuation value as discounted expected payoff
            continuation = np.exp(-self.r * self.dt) * payoff[:, t + 1]

            # Optimal strategy: max(immediate exercise, continuation)
            payoff[:, t] = np.maximum(immediate_exercise, continuation)

        # American call option price is the expected payoff at t=0, discounted
        return payoff[:, 0].mean()

    def american_put_option(self):
        # Simulate stock price paths
        S = self.stock_price_simulation()

        # Initialize payoff array
        payoff = np.zeros((self.simulations, self.steps + 1))

        # Compute payoff at expiration (last step)
        payoff[:, -1] = np.maximum(self.K - S[:, -1], 0)

        # Work backwards through time
        for t in range(self.steps - 1, -1, -1):
            # Calculate immediate exercise value
            immediate_exercise = np.maximum(self.K - S[:, t], 0)

            # Identify in-the-money paths
            itm = immediate_exercise > 0

            # Continuation value as discounted expected payoff
            continuation = np.exp(-self.r * self.dt) * payoff[:, t + 1]

            # Optimal strategy: max(immediate exercise, continuation)
            payoff[:, t] = np.maximum(immediate_exercise, continuation)

        # American put option price is the expected payoff at t=0, discounted
        return payoff[:, 0].mean()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate/european', methods=['POST'])
def calculate_european():
    try:
        # Get parameters from the form
        S0 = float(request.form.get('S0'))
        E = float(request.form.get('E'))
        T = float(request.form.get('T'))
        rf = float(request.form.get('rf'))
        sigma = float(request.form.get('sigma'))
        iterations = int(request.form.get('iterations'))

        # Create pricing model using the provided class
        model = EuropeanOptionPricing(S0, E, T, rf, sigma, iterations)

        # Calculate option prices
        call_price = model.call_option_simulation()
        put_price = model.put_option_simulation()

        return jsonify({
            'success': True,
            'call_price': round(call_price, 4),
            'put_price': round(put_price, 4)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/calculate/american', methods=['POST'])
def calculate_american():
    try:
        # Get parameters from the form
        S0 = float(request.form.get('S0'))
        K = float(request.form.get('K'))
        T = float(request.form.get('T'))
        r = float(request.form.get('r'))
        sigma = float(request.form.get('sigma'))
        steps = int(request.form.get('steps'))
        simulations = int(request.form.get('simulations'))

        # Create pricing model
        model = AmericanOptionPricing(S0, K, T, r, sigma, steps, simulations)

        # Calculate option prices
        call_price = model.american_call_option()
        put_price = model.american_put_option()

        return jsonify({
            'success': True,
            'call_price': round(call_price, 4),
            'put_price': round(put_price, 4)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    # app.run(debug=True)
    pass


def handler(event, context):
    with app.request_context(event):
        return app.full_dispatch_request()