import numpy as np
import coastlib.math.derivatives
import mpmath


distributions = [
    'genextreme',
    'genpareto'
]


class GeneralizedExtreme:
    """
    Generalized Extreme Value (GEV) distribution.
    https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    Has methods equivalent to those of scipy.stats.genextreme.
    Shape parameter has sign opposite to that of scipy.
    Shape parameter is called 'shape', while in scipy its called 'c'.
    """

    def __init__(self):
        pass

    @staticmethod
    def check_support(x, shape, loc, scale, dx='1e-10', precision=100):
        """
        Verify that passed parameters are valid.
        Returns np.nan in place of invalid inputs (if array, replaces invalid values with np.nan's).
        """

        with mpmath.workdps(precision):
            if not np.isscalar(x):
                x = np.array(x)

            # Parameters constraint
            if scale <= 0:
                raise ValueError('Scale parameter must be larger than 0')

            # Support constraint
            if np.isscalar(x):
                # Shape > 0
                if mpmath.mpf(shape) > mpmath.mpf(dx):
                    condition = x >= loc - scale / shape
                # Shape < 0
                elif mpmath.mpf(shape) < -mpmath.mpf(dx):
                    condition = x <= loc - scale / shape
                # Shape == 0
                else:
                    condition = True
            else:
                # Shape > 0
                if mpmath.mpf(shape) > mpmath.mpf(dx):
                    condition = np.all(x >= loc - scale / shape)
                # Shape < 0
                elif mpmath.mpf(shape) < -mpmath.mpf(dx):
                    condition = np.all(x <= loc - scale / shape)
                # Shape == 0
                else:
                    condition = True

            # Test conditions - replace invalid x's with np.nan
            if condition:
                return x
            else:
                if np.isscalar(x):
                    return np.nan
                else:
                    truncated_x = x
                    # Shape > 0
                    if mpmath.mpf(shape) > mpmath.mpf(dx):
                        truncated_x[truncated_x < loc - scale / shape] = np.nan
                    # Shape < 0
                    elif mpmath.mpf(shape) < -mpmath.mpf(dx):
                        truncated_x[truncated_x > loc - scale / shape] = np.nan
                    # Shape == 0
                    else:
                        pass
                    return truncated_x

    def rvs(self, shape, loc, scale, size=1, random_state=None, dx='1e-10', precision=100):
        """
        Returns random value or an array of random values of given size sampled from the GEV distribution.

        Parameters
        ----------
        shape : float
        loc : float
        scale : float
        size : int, optional
        random_state : int, optional
        dx : str, optional
        precision : int, optional
        """

        # Parameter constraint
        if scale <= 0:
            raise ValueError(f'Invalid parameter passed in scale={scale}')

        with mpmath.workdps(precision):
            if random_state is not None:
                np.random.seed(random_state)
            return self.ppf(np.random.random(size), shape, loc, scale, dx, precision)

    @staticmethod
    def fit(data, *args, **kwargs):
        """
        Returns shape parameter with a sign opposite to that of scipy.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html
        """

        raw_scipy = scipy.stats.genextreme.fit(np.float64(data), *args, **kwargs)
        return raw_scipy * np.array([-1, 1, 1])

    def __t(self, x, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates support value t, used further in self.cdf and self.pdf.

        Parameters
        ----------
        x : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        with mpmath.workdps(precision):
            # Make sure passed values are valid
            x = self.check_support(x, shape, loc, scale, dx, precision)

            # Shape == 0
            if mpmath.fabs(shape) <= mpmath.mpf(dx):
                if np.isscalar(x):
                    return mpmath.exp(-(mpmath.mpf(x) - mpmath.mpf(loc)) / mpmath.mpf(scale))
                else:
                    return np.array(
                        [
                            mpmath.exp(-(mpmath.mpf(_x) - mpmath.mpf(loc)) / mpmath.mpf(scale))
                            for _x in x
                        ]
                    )
            # Shape != 0
            else:
                if np.isscalar(x):
                    part_1 = mpmath.mpf('1') +\
                             mpmath.mpf(shape) * (mpmath.mpf(x) - mpmath.mpf(loc)) / mpmath.mpf(scale)
                    part_2 = -mpmath.mpf('1') / mpmath.mpf(shape)
                    return mpmath.power(part_1, part_2)
                else:
                    part_1 = np.array(
                        [
                            mpmath.mpf('1') +
                            mpmath.mpf(shape) * (mpmath.mpf(_x) - mpmath.mpf(loc)) / mpmath.mpf(scale)
                            for _x in x
                        ]
                    )
                    part_2 = -mpmath.mpf('1') / mpmath.mpf(shape)
                    return np.array(
                        [
                            mpmath.power(p_1, part_2) for p_1 in part_1
                        ]
                    )

    @staticmethod
    def ppf(q, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates percent point function (inverse CDF, value for given lower tail probability).

        Parameters
        ----------
        q : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        # Parameter constraint
        if scale <= 0:
            raise ValueError(f'Invalid parameter passed in scale={scale}')

        with mpmath.workdps(precision):
            # Shape == 0
            if mpmath.fabs(shape) <= mpmath.mpf(dx):
                if np.isscalar(q):
                    if q <= 0 or q >= 1:
                        raise ValueError(f'Quantile must lie in interval (0;1), {q} was given')
                    return mpmath.mpf(loc) - mpmath.mpf(scale) * mpmath.log(-mpmath.log(mpmath.mpf(q)))
                else:
                    for _q in q:
                        if _q <= 0 or _q >= 1:
                            raise ValueError(f'Quantile must lie in interval (0;1), {_q} was given')
                    return np.array(
                        [
                            mpmath.mpf(loc) - mpmath.mpf(scale) * mpmath.log(-mpmath.log(mpmath.mpf(_q)))
                            for _q in q
                        ]
                    )
            # Shape != 0
            else:
                if np.isscalar(q):
                    # Shape > 0
                    if mpmath.mpf(shape) > mpmath.mpf(dx) and (q < 0 or q >= 1):
                        raise ValueError(f'Quantile must lie in interval [0;1), {q} was given')
                    # Shape < 0
                    if mpmath.mpf(shape) < -mpmath.mpf(dx) and (q <= 0 or q > 1):
                        raise ValueError(f'Quantile must lie in interval (0;1], {q} was given')
                    part = mpmath.power(
                        -mpmath.log(mpmath.mpf(q)), -mpmath.mpf(shape)
                    ) - mpmath.mpf('1')
                    return mpmath.mpf(loc) + mpmath.mpf(scale) * part / mpmath.mpf(shape)
                else:
                    for _q in q:
                        # Shape > 0
                        if mpmath.mpf(shape) > mpmath.mpf(dx) and (_q < 0 or _q >= 1):
                            raise ValueError(f'Quantile must lie in interval [0;1), {_q} was given')
                        # Shape < 0
                        if mpmath.mpf(shape) < -mpmath.mpf(dx) and (_q <= 0 or _q > 1):
                            raise ValueError(f'Quantile must lie in interval (0;1], {_q} was given')
                    parts = np.array(
                        [
                            mpmath.power(
                                -mpmath.log(mpmath.mpf(_q)), -mpmath.mpf(shape)
                            ) - mpmath.mpf('1')
                            for _q in q
                        ]
                    )
                    return np.array(
                        [
                            mpmath.mpf(loc) + mpmath.mpf(scale) * part / mpmath.mpf(shape)
                            for part in parts
                        ]
                    )

    def isf(self, q, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates inverse survival function (inverse SF, value for given upper tail probability).

        Parameters
        ----------
        q : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        with mpmath.workdps(precision):
            return self.ppf(1-q, shape, loc, scale, dx, precision)

    def pdf(self, x, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates pdf.

        Parameters
        ----------
        x : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        with mpmath.workdps(precision):
            # Make sure passed values are valid
            x = self.check_support(x, shape, loc, scale, dx, precision)

            part_1 = mpmath.mpf('1') / mpmath.mpf(scale)
            if np.isscalar(x):
                part_2 = mpmath.power(
                    self.__t(x, shape, loc, scale, dx, precision),
                    mpmath.mpf(shape) + mpmath.mpf('1')
                )
                part_3 = mpmath.exp(-self.__t(x, shape, loc, scale, dx, precision))
                return part_1 * part_2 * part_3
            else:
                part_2 = np.array(
                    [
                        mpmath.power(
                            self.__t(_x, shape, loc, scale, dx, precision),
                            mpmath.mpf(shape) + mpmath.mpf('1')
                        ) for _x in x
                    ]
                )
                part_3 = np.array(
                    [
                        mpmath.exp(
                            -self.__t(_x, shape, loc, scale, dx, precision)
                        ) for _x in x
                    ]
                )
                return np.array(
                    [part_1 * p_2 * p_3 for p_2, p_3 in zip(part_2, part_3)]
                )

    def cdf(self, x, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates cdf.

        Parameters
        ----------
        x : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        with mpmath.workdps(precision):
            # Make sure passed values are valid
            x = self.check_support(x, shape, loc, scale, dx, precision)

            if np.isscalar(x):
                return mpmath.exp(-self.__t(x, shape, loc, scale, dx, precision))
            else:
                return np.array(
                    [
                        mpmath.exp(-self.__t(_x, shape, loc, scale, dx, precision))
                        for _x in x
                    ]
                )

    def log_likelihood(self, data, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates log-likelihood.

        Parameters
        ----------
        data : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        with mpmath.workdps(precision):
            # Make sure passed values are valid
            data = self.check_support(data, shape, loc, scale, dx, precision)

            if np.isscalar(data):
                return mpmath.log(self.pdf(data, shape, loc, scale, dx, precision))
            else:
                return mpmath.fsum(
                    [
                        mpmath.log(
                            self.pdf(_x, shape, loc, scale, dx, precision)
                        ) for _x in data
                    ]
                )

    def observed_information(self, data, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates observed (Fisher) information matrix for 3-parameter GEV (non-degenerate).
        Calculate observed_information manually if any of the parameters are fixed (were assumed and not estimated).

        Parameters
        ----------
        data : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        with mpmath.workdps(precision):
            # Make sure passed values are valid
            data = self.check_support(data, shape, loc, scale, dx, precision)

            # Define log likelihood function to be differentiated
            def log_likelihood_stationary(*theta):
                return self.log_likelihood(data, theta[0], theta[1], theta[2], dx, precision)

            return -coastlib.math.derivatives.hessian(
                func=log_likelihood_stationary, n=3,
                coordinates=[shape, loc, scale], dx=dx, precision=precision
            )


genextreme = GeneralizedExtreme()


class GeneralizedPareto:
    """
    Generalized Pareto distribution (GPD).
    https://en.wikipedia.org/wiki/Generalized_Pareto_distribution
    Has methods equivalent to those of scipy.stats.genepareto.
    Shape parameter is called 'shape', while in scipy its called 'c'.
    """

    def __init__(self):
        pass

    @staticmethod
    def check_support(x, shape, loc, scale, dx='1e-10', precision=100):
        """
        Verify that passed parameters are valid.
        Returns np.nan in place of invalid inputs (if array, replaces invalid values with np.nan's).
        """

        with mpmath.workdps(precision):
            if not np.isscalar(x):
                x = np.array(x)

            # Parameters constraint
            if scale <= 0:
                raise ValueError('Scale parameter must be larger than 0')

            # Support constraint
            if np.isscalar(x):
                # Shape >= 0
                if mpmath.mpf(shape) >= -mpmath.mpf(dx):
                    condition = x >= loc
                # Shape < 0
                else:
                    condition = loc <= x <= loc - scale / shape
            else:
                # Shape >= 0
                if mpmath.mpf(shape) >= -mpmath.mpf(dx):
                    condition = np.all(x >= loc)
                # Shape < 0
                else:
                    condition = np.all(x >= loc) and np.all(x <= loc - scale / shape)

            # Test conditions - replace invalid x's with np.nan
            if condition:
                return x
            else:
                if np.isscalar(x):
                    return np.nan
                else:
                    truncated_x = x
                    # Shape >= 0
                    if mpmath.mpf(shape) >= -mpmath.mpf(dx):
                        truncated_x[truncated_x < loc] = np.nan
                    # Shape < 0
                    else:
                        truncated_x[truncated_x < loc] = np.nan
                        truncated_x[truncated_x > loc - scale / shape] = np.nan
                    return truncated_x

    def rvs(self, shape, loc, scale, size=1, random_state=None, dx='1e-10', precision=100):
        """
        Returns random value or an array of random values of given size sampled from the GPD distribution.

        Parameters
        ----------
        shape : float
        loc : float
        scale : float
        size : int, optional
        random_state : int, optional
        dx : str, optional
        precision : int, optional
        """

        # Parameter constraint
        if scale <= 0:
            raise ValueError('Scale parameter must be larger than 0')

        with mpmath.workdps(precision):
            if random_state is not None:
                np.random.seed(random_state)
            return self.ppf(np.random.random(size), shape, loc, scale, dx, precision)

    @staticmethod
    def fit(data, *args, **kwargs):
        """
        Returns shape parameter with a sign opposite to that of scipy.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html
        """

        return scipy.stats.genpareto.fit(data, *args, **kwargs)

    def __z(self, x, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates support value z, used further in self.cdf and self.pdf.

        Parameters
        ----------
        x : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        with mpmath.workdps(precision):
            # Make sure passed values are valid
            x = self.check_support(x, shape, loc, scale, dx, precision)

            if np.isscalar(x):
                return (x - loc) / scale
            else:
                return np.array(
                    [
                        (_x - loc) / scale
                        for _x in x
                    ]
                )

    @staticmethod
    def ppf(q, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates percent point function (inverse CDF, value for given lower tail probability).

        Parameters
        ----------
        q : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        # Parameter constraint
        if scale <= 0:
            raise ValueError(f'Invalid parameter passed in scale={scale}')

        with mpmath.workdps(precision):
            # Shape == 0
            if mpmath.fabs(shape) <= mpmath.mpf(dx):
                if np.isscalar(q):
                    if q <= 0 or q > 1:
                        raise ValueError(f'Quantile must lie in interval (0;1], {q} was given')
                    part_1 = mpmath.mpf('1') - mpmath.mpf(q)
                    part_2 = mpmath.log(part_1)
                    return -part_2 * mpmath.mpf(scale) + mpmath.mpf(loc)
                else:
                    for _q in q:
                        if _q <= 0 or _q > 1:
                            raise ValueError(f'Quantile must lie in interval (0;1], {_q} was given')
                    parts_1 = [
                        mpmath.mpf('1') - mpmath.mpf(_q)
                        for _q in q
                    ]
                    parts_2 = [mpmath.log(part_1) for part_1 in parts_1]
                    return np.array(
                        [
                            -part_2 * mpmath.mpf(scale) + mpmath.mpf(loc)
                            for part_2 in parts_2
                        ]
                    )
            # Shape != 0
            else:
                if np.isscalar(q):
                    if q <= 0 or q > 1:
                        raise ValueError(f'Quantile must lie in interval (0;1], {q} was given')
                    part_1 = mpmath.mpf('1') - mpmath.mpf(q)
                    part_2 = -mpmath.mpf('1') / mpmath.mpf(shape)
                    part_3 = mpmath.power(part_1, mpmath.mpf('1') / part_2) - mpmath.mpf('1')
                    return part_3 * mpmath.mpf(scale) / mpmath.mpf(shape) + mpmath.mpf(loc)
                else:
                    for _q in q:
                        if _q <= 0 or _q > 1:
                            raise ValueError(f'Quantile must lie in interval (0;1], {_q} was given')
                    parts_1 = [
                        mpmath.mpf('1') - mpmath.mpf(_q)
                        for _q in q
                    ]
                    part_2 = -mpmath.mpf('1') / mpmath.mpf(shape)
                    parts_3 = [
                        mpmath.power(part_1, mpmath.mpf('1') / part_2) - mpmath.mpf('1')
                        for part_1 in parts_1
                    ]
                    return np.array(
                        [
                            part_3 * mpmath.mpf(scale) / mpmath.mpf(shape) + mpmath.mpf(loc)
                            for part_3 in parts_3
                        ]
                    )

    def isf(self, q, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates inverse survival function (inverse SF, value for given upper tail probability).

        Parameters
        ----------
        q : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        with mpmath.workdps(precision):
            return self.ppf(1-q, shape, loc, scale, dx, precision)

    def pdf(self, x, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates pdf.

        Parameters
        ----------
        x : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        with mpmath.workdps(precision):
            # Make sure passed values are valid
            x = self.check_support(x, shape, loc, scale, dx, precision)
            z = self.__z(x, shape, loc, scale, dx, precision)

            part_1 = mpmath.mpf('1') / mpmath.mpf(scale)
            part_3 = -(mpmath.mpf('1') / mpmath.mpf(shape) + mpmath.mpf('1'))
            if np.isscalar(x):
                part_2 = mpmath.mpf('1') + mpmath.mpf(shape) * z
                return part_1 * mpmath.power(part_2, part_3)
            else:
                parts_2 = [
                    mpmath.mpf('1') + mpmath.mpf(shape) * _z
                    for _z in z
                ]
                return np.array(
                    [
                        part_1 * mpmath.power(part_2, part_3)
                        for part_2 in parts_2
                    ]
                )

    def cdf(self, x, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates cdf.

        Parameters
        ----------
        x : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        with mpmath.workdps(precision):
            # Make sure passed values are valid
            x = self.check_support(x, shape, loc, scale, dx, precision)
            z = self.__z(x, shape, loc, scale, dx, precision)

            part_2 = -(mpmath.mpf('1') / mpmath.mpf(shape))
            if np.isscalar(x):
                part_1 = mpmath.mpf('1') + mpmath.mpf(shape) * z
                return mpmath.mpf('1') - mpmath.power(part_1, part_2)
            else:
                parts_1 = [
                    mpmath.mpf('1') + mpmath.mpf(shape) * _z
                    for _z in z
                ]
                return np.array(
                    [
                        mpmath.mpf('1') - mpmath.power(part_1, part_2)
                        for part_1 in parts_1
                    ]
                )

    def log_likelihood(self, data, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates log-likelihood.

        Parameters
        ----------
        data : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        with mpmath.workdps(precision):
            # Make sure passed values are valid
            data = self.check_support(data, shape, loc, scale, dx, precision)

            if np.isscalar(data):
                return mpmath.log(self.pdf(data, shape, loc, scale, dx, precision))
            else:
                return mpmath.fsum(
                    [
                        mpmath.log(
                            self.pdf(_x, shape, loc, scale, dx, precision)
                        ) for _x in data
                    ]
                )

    def observed_information(self, data, shape, loc, scale, dx='1e-10', precision=100):
        """
        Calculates observed (Fisher) information matrix for 3-parameter GPD (non-degenerate).
        Calculate observed_information manually if any of the parameters are fixed (were assumed and not estimated).

        Parameters
        ----------
        data : float or array_like
        shape : float
        loc : float
        scale : float
        dx : str, optional
        precision : int, optional
        """

        with mpmath.workdps(precision):
            # Make sure passed values are valid
            data = self.check_support(data, shape, loc, scale, dx, precision)

            # Define log likelihood function to be differentiated
            def log_likelihood_stationary(*theta):
                return self.log_likelihood(data, theta[0], theta[1], theta[2], dx, precision)

            return -coastlib.math.derivatives.hessian(
                func=log_likelihood_stationary, n=3,
                coordinates=[shape, loc, scale], dx=dx, precision=precision
            )


genpareto = GeneralizedPareto()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.stats

    def __test_genextreme():
        np.random.seed(0)
        random_sample = genextreme.rvs(shape=.07, loc=1, scale=2, size=100)
        random_sample = np.sort(np.float64(random_sample))
        fit_parameters = genextreme.fit(random_sample, loc=1)

        scipy_genextreme = getattr(scipy.stats, 'genextreme')

        # Test pdf
        with plt.style.context('bmh'):
            plt.figure()
            plt.hist(
                random_sample, bins=15, density=True, cumulative=False, label='random sample'
            )
            plt.plot(
                random_sample, scipy_genextreme.pdf(
                    random_sample, c=-fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2]
                ),
                color='orangered', zorder=10, label='scipy'
            )
            plt.scatter(
                random_sample, genextreme.pdf(random_sample, *fit_parameters, dx='1e-10', precision=100),
                color='k', s=10, zorder=15, label='custom'
            )
            plt.legend()
            plt.title('genextreme distribution pdf test\nAll points should be on the line')
            plt.show()

        # Test cdf
        with plt.style.context('bmh'):
            plt.figure()
            plt.hist(
                random_sample, bins=100, density=True, cumulative=True, label='random sample'
            )
            plt.plot(
                random_sample, scipy_genextreme.cdf(
                    random_sample, c=-fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2]
                ),
                color='orangered', zorder=10, label='scipy'
            )
            plt.scatter(
                random_sample, genextreme.cdf(random_sample, *fit_parameters),
                color='k', s=10, zorder=15, label='custom'
            )
            plt.legend()
            plt.title('genextreme distribution cdf test\nAll points should be on the line')
            plt.show()

        # Test ppf
        quantiles = np.arange(0.01, 1, .01)
        ppfs = genextreme.ppf(
            q=quantiles, shape=fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2],
            dx='1e-10', precision=100
        )
        scipy_ppfs = scipy_genextreme.ppf(
            q=quantiles, c=-fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2]
        )
        with plt.style.context('bmh'):
            plt.figure()
            plt.plot(quantiles, ppfs, color='orangered', label='custom', zorder=5)
            plt.scatter(quantiles, scipy_ppfs, color='royalblue', label='scipy', zorder=10)
            plt.legend()
            plt.title('genextreme distribution ppf test\nAll points should be on the line')
            plt.show()

        # Test isf
        quantiles = np.arange(0.01, 1, .01)
        isfs = genextreme.isf(
            q=quantiles, shape=fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2],
            dx='1e-10', precision=100
        )
        scipy_isfs = scipy_genextreme.isf(
            q=quantiles, c=-fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2]
        )
        with plt.style.context('bmh'):
            plt.figure()
            plt.plot(quantiles, isfs, color='orangered', label='custom', zorder=5)
            plt.scatter(quantiles, scipy_isfs, color='royalblue', label='scipy', zorder=10)
            plt.legend()
            plt.title('genextreme distribution isf test\nAll points should be on the line')
            plt.show()

        scipy_ll = scipy_genextreme.logpdf(
            random_sample, c=-fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2]
        ).sum()
        ll = np.float64(genextreme.log_likelihood(random_sample, *fit_parameters, dx='1e-10', precision=100))

        def log_likelihood_stationary(*theta):
            return scipy_genextreme.logpdf(
                random_sample, c=-theta[0], loc=theta[1], scale=theta[2]
            ).sum()

        scipy_oi = -coastlib.math.derivatives.hessian(
            func=log_likelihood_stationary, n=3,
            coordinates=fit_parameters, dx='1e-6', precision=None
        )

        oi = genextreme.observed_information(
            random_sample, *fit_parameters, dx='1e-10', precision=100
        ).astype(np.float64)

        print(f'\nLog-likelihood'
              f'\nscipy numeric'
              f'\n{np.round(scipy_ll, 2)}'
              f'\nmpmath'
              f'\n{np.round(ll, 2)}')

        print(f'\nObserved information'
              f'\nscipy numeric'
              f'\n{np.round(scipy_oi, 2)}'
              f'\nmpmath'
              f'\n{np.round(oi, 2)}')

    def __test_genpareto():
        np.random.seed(0)
        random_sample = genpareto.rvs(shape=0.2, loc=0, scale=.7, size=100)
        random_sample = np.sort(np.float64(random_sample))
        fit_parameters = genpareto.fit(random_sample, floc=0)

        scipy_genpareto = getattr(scipy.stats, 'genpareto')

        # Test pdf
        with plt.style.context('bmh'):
            plt.figure()
            plt.hist(
                random_sample, bins=20, density=True, cumulative=False, label='random sample'
            )
            plt.plot(
                random_sample, scipy_genpareto.pdf(
                    random_sample, c=fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2]
                ),
                color='orangered', zorder=10, label='scipy'
            )
            plt.scatter(
                random_sample, genpareto.pdf(random_sample, *fit_parameters),
                color='k', s=10, zorder=15, label='custom'
            )
            plt.legend()
            plt.title('genpareto distribution pdf test\nAll points should be on the line')
            plt.show()

        # Test cdf
        with plt.style.context('bmh'):
            plt.figure()
            plt.hist(
                random_sample, bins=100, density=True, cumulative=True, label='random sample'
            )
            plt.plot(
                random_sample, scipy_genpareto.cdf(
                    random_sample, c=fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2]
                ),
                color='orangered', zorder=10, label='scipy'
            )
            plt.scatter(
                random_sample, genpareto.cdf(random_sample, *fit_parameters),
                color='k', s=10, zorder=15, label='custom'
            )
            plt.legend()
            plt.title('genpareto distribution cdf test\nAll points should be on the line')
            plt.show()

        # Test ppf
        quantiles = np.arange(0.01, 1, .01)
        ppfs = genpareto.ppf(
            q=quantiles, shape=fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2],
            dx='1e-10', precision=100
        )
        scipy_ppfs = scipy_genpareto.ppf(
            q=quantiles, c=fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2]
        )
        with plt.style.context('bmh'):
            plt.figure()
            plt.plot(quantiles, ppfs, color='orangered', label='custom', zorder=5)
            plt.scatter(quantiles, scipy_ppfs, color='royalblue', label='scipy', zorder=10)
            plt.legend()
            plt.title('genepareto distribution ppf test\nAll points should be on the line')
            plt.show()

        # Test isf
        quantiles = np.arange(0.01, 1, .01)
        isfs = genpareto.isf(
            q=quantiles, shape=fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2],
            dx='1e-10', precision=100
        )
        scipy_isfs = scipy_genpareto.isf(
            q=quantiles, c=fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2]
        )
        with plt.style.context('bmh'):
            plt.figure()
            plt.plot(quantiles, isfs, color='orangered', label='custom', zorder=5)
            plt.scatter(quantiles, scipy_isfs, color='royalblue', label='scipy', zorder=10)
            plt.legend()
            plt.title('genextreme distribution isf test\nAll points should be on the line')
            plt.show()

        scipy_ll = scipy_genpareto.logpdf(
            random_sample, c=fit_parameters[0], loc=fit_parameters[1], scale=fit_parameters[2]
        ).sum()
        ll = np.float64(genpareto.log_likelihood(random_sample, *fit_parameters))

        def log_likelihood_stationary(*theta):
            return scipy_genpareto.logpdf(
                random_sample, c=theta[0], loc=theta[1], scale=theta[2]
            ).sum()

        scipy_oi = -coastlib.math.derivatives.hessian(
            func=log_likelihood_stationary, n=3,
            coordinates=fit_parameters, dx='1e-6', precision=None
        )
        oi = genpareto.observed_information(
            random_sample, *fit_parameters, dx='1e-10', precision=100
        ).astype(np.float64)

        print(f'\nLog-likelihood'
              f'\nscipy numeric'
              f'\n{np.round(scipy_ll, 2)}'
              f'\nmpmath'
              f'\n{np.round(ll, 2)}')

        print(f'\nObserved information'
              f'\nscipy numeric'
              f'\n{np.round(scipy_oi, 2)}'
              f'\nmpmath'
              f'\n{np.round(oi, 2)}')

    __test_genextreme()
    __test_genpareto()
