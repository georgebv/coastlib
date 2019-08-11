# coastlib, a coastal engineering Python library
# Copyright (C), 2019 Georgii Bocharov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import scipy.stats
import mpmath


def partial_derivative(func, var, order=1, coordinates=None, dx='1e-6', precision=100):
    """
    Generates a function which returns estimates of partial derivatives of <func> by argument with index <var>
    using the central difference formula.
    If <coordinates> are passed, provides a numerical estimate of the partial derivative. This is equivalent
    to passing <coordinates> to a generated function with <coordinates=None>
    https://en.wikipedia.org/wiki/Partial_derivative

    Parameters
    ----------
    func : function
        Function, partial derivative of which is estimated. Takes n positional arguments X1...Xn
    var : int
        Index of argument by which <func> is differentiated.
        Each value should be in the range [0, len(<coordinates>)-1]
    order : int, optional
        Order of derivative (default=1)
    coordinates : array_like or float, optional
        List of n coordinates X1...Xn at which the <func> derivative is estimated (default=None)
    dx : str, optional
        String representing a float representing spacing at which the partial derivative is estimated (default='1e-6')
    precision : int, optional
        Precision of floating point calculations (see mpmath library documentation) (default=100)
        Set to None to run with numpy without mpmath
        Some functions are incompatible with mpmath. Set <precision> to None to avoid errors or rewrite the function.
        Derivative estimated without high <precision> may have a significant error due to rounding and under-/overflow

    Returns
    -------
    function or float/mpmath.mpf (mpmath.mpf if <precision> is not None)
        Function estimating or numerical estimate of partial derivative of the <func> evaluated at <coordinates>
    """

    if not isinstance(order, int) or order < 1:
        raise RuntimeError(f'Bad value passed in order={order}')

    # Convert inputs to mpmath real float objects
    if precision is not None:
        with mpmath.workdps(precision):
            if coordinates is not None:
                if np.isscalar(coordinates):
                    coordinates = [mpmath.mpf(str(coordinates))]
                else:
                    coordinates = [mpmath.mpf(str(_c)) for _c in coordinates]

            # Define numerical partial derivative function using the central difference formula
            def pdf(*args):
                args_forward = np.append(np.append(args[:var], [args[var] + mpmath.mpf(dx)]), args[var + 1:])
                args_backward = np.append(np.append(args[:var], [args[var] - mpmath.mpf(dx)]), args[var + 1:])
                return (func(*args_forward) - func(*args_backward)) / (2 * mpmath.mpf(dx))

            # Estimate partial derivative if coordinates were provided
            if order == 1:
                if coordinates is not None:
                    return pdf(*coordinates)
                else:
                    return pdf
            else:
                return partial_derivative(
                    func=pdf, var=var, order=order-1, coordinates=coordinates, dx=dx, precision=precision
                )
    else:
        # Define numerical partial derivative function using the central difference formula
        def pdf(*args):
            args_forward = np.append(np.append(args[:var], [args[var] + float(dx)]), args[var + 1:])
            args_backward = np.append(np.append(args[:var], [args[var] - float(dx)]), args[var + 1:])
            return (func(*args_forward) - func(*args_backward)) / (2 * float(dx))

        # Estimate partial derivative if coordinates were provided
        if order == 1:
            if coordinates is not None:
                return pdf(*coordinates)
            else:
                return pdf
        else:
            return partial_derivative(
                func=pdf, var=var, order=order-1, coordinates=coordinates, dx=dx, precision=precision
            )


def gradient(func, n, coordinates=None, dx='1e-6', precision=100):
    """
    Estimates gradient of the <func> at given <coordinates>
    or returns an array with partial derivative functions of shape (n,1)
    https://en.wikipedia.org/wiki/Gradient

    Parameters
    ----------
    func : function
        Function, gradient of which is estimated. Takes <n> positional arguments X1...Xn
    n : int
        Number of positional arguments <func> takes
    coordinates : array_like or float, optional
        List of n coordinates X1...Xn, at which the <func> gradient is estimated (default=None)
        Must satisfy len(<coordinates>)==<n>
    dx : str, optional
        String representing a float representing spacing at which the partial derivatives are estimated (default='1e-6')
    precision : int, optional
        Precision of floating point calculations (see mpmath library documentation) (default=100)
        Some functions are incompatible with mpmath. Set <precision> to None to avoid errors or rewrite the function.
        Derivative estimated without high <precision> may have a significant error due to rounding and under-/overflow

    Returns
    -------
    numpy.ndarray with functions or float/mpmath.mpf's
        The gradient matrix of the <func> with either partial derivative functions or estimates
        of these functions at <coordinates>
    """
    return np.array(
        [
            [partial_derivative(func=func, var=i, order=1, coordinates=coordinates, dx=dx, precision=precision)]
            for i in range(n)
        ]
    )


def hessian(func, n, coordinates=None, dx='1e-6', precision=100):
    """
    Estimates Hessian of the <base_function> at given <coordinates>
    or returns an array with respective partial derivative functions of shape (n,n)
    https://en.wikipedia.org/wiki/Hessian

    Parameters
    ----------
    func : function
        Function, Hessian of which is estimated. Takes <n> positional arguments X1...Xn
    n : int
        Number of positional arguments <func> takes
    coordinates : array_like or float, optional
        List of n coordinates X1...Xn, at which the <func> Hessian is estimated.
        Must satisfy len(<coordinates>)==<n> (default=None)
    dx : str, optional
        String representing a float representing spacing at which the partial derivatives are estimated (default='1e-6')
    precision : int, optional
        Precision of floating point calculations (see mpmath library documentation) (default=100)
        Some functions are incompatible with mpmath. Set <precision> to None to avoid errors or rewrite the function.
        Derivative estimated without high <precision> may have a significant error due to rounding and under-/overflow

    Returns
    -------
    numpy.ndarray with functions or float/mpmath.mpf's
        The Hessian matrix of the <func> with either partial derivative functions or estimates
        of these functions at <coordinates>
    """
    hessian_matrix = []
    for i in range(n):
        partial_derivative_i = partial_derivative(
            func=func, var=i, order=1, coordinates=None, dx=dx, precision=precision
        )
        hessian_matrix_i = []
        for j in range(n):
            partial_derivative_ij = partial_derivative(
                func=partial_derivative_i, var=j, order=1, coordinates=coordinates, dx=dx, precision=precision
            )
            hessian_matrix_i.append(partial_derivative_ij)
        hessian_matrix.append(hessian_matrix_i)
    return np.array(hessian_matrix)


def delta_confidence(x, scalar_function, likelihood_function, theta, alpha=0.95, dx='1e-6', precision=100):
    """
    Estimates confidence interval of <scalar_function> for value(s) <x> with probability <alpha>,
    assuming asymptotic normality of <mle> estimates and resulting <scalar_function> outputs for same <x>.
    <mle> is(are) estimate(s) of parameters maximizing the <likelihood_function>.
    https://en.wikipedia.org/wiki/Delta_method

    Parameters
    ----------
    x : array_like or float
        Scalar value(s) taken by <scalar_function> for which the confidence interval(s) is(are) estimated
    scalar_function : function
        Function which takes <x> and <mle> as its arguments <scalar_function(x, *mle)> and returns a scalar value.
        Takes <x> and <*mle> as its arguments
        Can be a return value function estimating return value for a specific return period (constant)
        using a distribution with parameters <mle>
    likelihood_function : function
        Likelihood function, by maximizing parameters of which the <mle> was(were) estimated
        Takes <*mle> as its arguments
        Can be the log-likelihood function of a distribution with parameter(s) <mle>
    theta : array_like or float
        Array or float with parameter(s) estimated by maximizing the <likelihood_function>
        Can be an output of scipy.stats.some_distribution.fit(some_data)
    alpha : float, optional
        Probability that a <scalar_function> will be drawn from the returned range of normal distribution
        with <scalar_function(*mle)> as location and <scalar_function> variance as scale.
        Value should be in the range [0, 1]
    dx : str, optional
        String representing a float representing spacing at which the partial derivatives are estimated (default='1e-6')
    precision : int, optional
        Precision of floating point calculations (see mpmath library documentation) (default=100)
        Some functions are incompatible with mpmath. Set <precision> to None to avoid errors or rewrite the functions.
        Derivative estimated without high <precision> may have a significant error due to rounding and under-/overflow

    Returns
    -------
    numpy.ndarray of shape (2,) of floats with lower and upper confidence bounds
    or numpy.ndarray of shape (2,1) with 2 arrays of floats for lower and upper confidence bounds for each <x>
        end-points of range that contains 100*<alpha>% of the <scalar_function> possible values
    """
    if np.isscalar(x):
        def __scalar_function(*coordinates):
            return scalar_function(x, *coordinates)
        location = scalar_function(*theta)
        delta_scalar = gradient(func=__scalar_function, n=len(theta), coordinates=theta, dx=dx, precision=precision)
        information_matrix = -hessian(
            func=likelihood_function, n=len(theta),
            coordinates=theta, dx=dx, precision=precision
        )
        scale = np.dot(np.dot(delta_scalar.T, np.linalg.inv(information_matrix)), delta_scalar).flatten()[0]
        return scipy.stats.norm.interval(alpha=alpha, loc=location, scale=scale)
    else:
        locations, scales = [], []
        for _x in x:
            def __scalar_function(*coordinates):
                return scalar_function(_x, *coordinates)
            locations.append(scalar_function(_x, *theta))
            delta_scalar = gradient(
                func=__scalar_function, n=len(theta),
                coordinates=theta, dx=dx, precision=precision
            )
            information_matrix = -hessian(
                func=likelihood_function, n=len(theta),
                coordinates=theta, dx=dx, precision=precision
            )
            scales.append(
                np.dot(np.dot(delta_scalar.T, np.linalg.inv(information_matrix)), delta_scalar).flatten()[0]
            )
        return np.array(
            [
                scipy.stats.norm.interval(alpha=alpha, loc=location, scale=scale)
                for location, scale in zip(locations, scales)
            ]
        ).T


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # region Define test functions

    def __test_gradient():

        def test_function(x, y):
            return mpmath.sin(x) * y ** 2

        def analytical_gradient(x, y):
            return np.array([[np.cos(x) * y ** 2], [2 * np.sin(x) * y]])

        test_coordinates = np.array([1000, 2])
        estimate = gradient(func=test_function, n=2, coordinates=test_coordinates, dx='1e-10', precision=100)
        analytical = analytical_gradient(*test_coordinates)
        print(f'\n'
              f'Estimate\n{estimate.astype(np.float64)}\n'
              f'Analytical\n{analytical.astype(np.float64)}')
        if np.allclose(
                estimate.astype(np.float64),
                analytical.astype(np.float64),
                rtol=1e-6, atol=1e-6
        ):
            print('Test passed')
        else:
            print('Test failed')


    def __test_hessian():

        def test_function(x, y):
            return mpmath.sin(x) * y ** 2

        def analytical_hessian(x, y):
            return np.array(
                [
                    [-np.sin(x) * y ** 2, 2 * np.cos(x) * y],
                    [2 * np.cos(x) * y, 2 * np.sin(x)]
                ]
            )

        test_coordinates = np.array([1, 1])
        estimate = hessian(func=test_function, n=2, coordinates=test_coordinates, dx='1e-10', precision=100)
        analytical = analytical_hessian(*test_coordinates)
        print(f'\n'
              f'Estimate\n{estimate.astype(np.float64)}\n'
              f'Analytical\n{analytical}')
        if np.allclose(
                estimate.astype(np.float64),
                analytical,
                rtol=1e-6, atol=1e-6
        ):
            print('Test passed')
        else:
            print('Test failed')


    def __test_delta():
        np.random.seed(0)
        distribution = scipy.stats.genextreme
        random_sample = distribution.rvs(-0.1, loc=7, scale=.3, size=15)
        mle_estimate = distribution.fit(random_sample)
        eval_range = np.linspace(random_sample.min(), random_sample.max(), 100)

        def likelihood_function(*mle):
            return np.sum(np.log(distribution.pdf(random_sample, *mle)))

        def scalar_function(x, *mle):
            return distribution.pdf(x, *mle)

        confidence_intervals = delta_confidence(
            x=eval_range, scalar_function=scalar_function, likelihood_function=likelihood_function,
            theta=mle_estimate, alpha=0.95, dx='1e-6', precision=None
        )
        conf_bot, conf_top = confidence_intervals[0], confidence_intervals[1]

        with plt.style.context('bmh'):
            plt.hist(
                random_sample, density=True, cumulative=False, bins=10, zorder=0, rwidth=0.9
            )
            plt.plot(eval_range, distribution.pdf(eval_range, *mle_estimate), zorder=10, color='orangered')
            plt.fill_between(eval_range, conf_bot, conf_top, alpha=0.7, zorder=5, color='k')
            plt.show()
    # endregion
    __test_gradient()
    __test_hessian()
    __test_delta()
