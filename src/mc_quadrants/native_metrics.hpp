// Numerical reporting kernels; output shaping remains at the Python boundary.
extern "C" int mc_money_weighted_returns(
    int periods, int paths, double initial, double contribution,
    const double *terminal, double *output
) {
    if (periods <= 0 || paths <= 0 || initial <= 0.0 || contribution < 0.0 || !terminal || !output) return 1;
    parallel_paths(paths, 8, [&](int begin, int end) {
        for (int path = begin; path < end; ++path) {
            double low = -0.99, high = 10.0;
            for (int iteration = 0; iteration < 64; ++iteration) {
                const double rate = (low + high) / 2.0;
                // Geometric annuity replaces the periods x paths discount matrix.
                // Scale the NPV by the terminal discount for negative rates to
                // avoid overflow; expm1 keeps near-zero rates well conditioned.
                const double exponent = periods * std::log1p(rate);
                double npv;
                if (rate < 0.0) {
                    const double growth = std::exp(exponent);
                    const double annuity = std::expm1(exponent) / rate;
                    npv = -initial * growth - contribution * annuity + terminal[path];
                } else {
                    const double discount = std::exp(-exponent);
                    const double annuity = rate == 0.0 ? periods : -std::expm1(-exponent) / rate;
                    npv = -initial - contribution * annuity + terminal[path] * discount;
                }
                if (npv > 0.0) low = rate; else high = rate;
            }
            output[path] = std::clamp(std::pow(1.0 + (low + high) / 2.0, 12.0) - 1.0, -1.0, 100.0);
        }
    });
    return 0;
}

extern "C" int mc_risk_path_statistics(
    int periods, int paths, double initial, const double *wealth,
    const double *contributions, int contribution_width,
    const double *withdrawals, int withdrawal_width,
    const double *risk_free, int risk_free_width, double *statistics
) {
    if (periods <= 0 || paths <= 0 || !wealth || !contributions || !withdrawals || !risk_free || !statistics ||
        (contribution_width != 1 && contribution_width != paths) ||
        (withdrawal_width != 1 && withdrawal_width != paths) ||
        (risk_free_width != 1 && risk_free_width != paths)) return 1;
    parallel_paths(paths, 8, [&](int begin, int end) {
        for (int path = begin; path < end; ++path) {
            double peak = initial, previous = initial, moments[9]{};
            for (int t = 0; t < periods; ++t) {
                const double value = wealth[static_cast<std::size_t>(t) * paths + path];
                peak = std::max(peak, value);
                const double drawdown = value / peak - 1.0;
                moments[7] = std::max(moments[7], -drawdown);
                moments[8] += drawdown * drawdown;
                const double denominator = previous + contributions[static_cast<std::size_t>(t) * contribution_width + (contribution_width == 1 ? 0 : path)];
                const double numerator = value + withdrawals[static_cast<std::size_t>(t) * withdrawal_width + (withdrawal_width == 1 ? 0 : path)];
                if (denominator > 0.0 && numerator >= 0.0) {
                    const double r = numerator / denominator - 1.0;
                    if (std::isfinite(r)) {
                        const double excess = r - risk_free[static_cast<std::size_t>(t) * risk_free_width + (risk_free_width == 1 ? 0 : path)];
                        moments[0] += r; moments[1] += r * r; moments[2] += 1.0;
                        if (r > -1.0) { moments[3] += std::log1p(r); moments[4] += 1.0; }
                        moments[5] += excess;
                        if (excess < 0.0) moments[6] += excess * excess;
                    }
                }
                previous = value;
            }
            moments[8] = std::sqrt(moments[8] / (periods + 1));
            for (int k = 0; k < 9; ++k) statistics[static_cast<std::size_t>(k) * paths + path] = moments[k];
        }
    });
    return 0;
}

extern "C" int mc_inflation_index(
    int periods, int paths, double frequency, const double *rates, int rate_width,
    double annual_rate, int inverse, double *output
) {
    if (periods <= 0 || paths <= 0 || !(frequency > 0.0) || !output ||
        (rates && rate_width != 1 && rate_width != paths)) return 1;
    const double constant = std::pow(1.0 + annual_rate, 1.0 / frequency);
    parallel_paths(paths, 8, [&](int begin, int end) {
        for (int path = begin; path < end; ++path) {
            double cumulative = 1.0;
            for (int t = 0; t < periods; ++t) {
                cumulative *= rates ? std::pow(1.0 + rates[static_cast<std::size_t>(t) * rate_width + (rate_width == 1 ? 0 : path)], 1.0 / frequency) : constant;
                output[static_cast<std::size_t>(t) * paths + path] = inverse ? 1.0 / cumulative : cumulative;
            }
        }
    });
    return 0;
}
