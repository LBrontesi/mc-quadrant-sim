// Included by native_simulation.cpp: tax-neutral portfolio path accounting.
// Python validates/normalizes configuration; all path/period work lives here.
struct MCNeutralPortfolioConfig {
    int periods, paths, assets, threads, mode, simple_returns, rebalance_frequency;
    int contribution_mode, guardrail_policy, skip_inflation_after_loss;
    double initial_value, contribution, leverage, financing_growth, maintenance_margin;
    double cost_rate, upper_guardrail, lower_guardrail, adjustment, floor, ceiling;
    const double *returns, *weights, *fee_logs, *cost_paths, *financing_paths, *cpi;
    const int *phase_ids;
    const double *phase_amounts, *due_factors, *one_times;
    const std::uint8_t *reviews;
};

extern "C" int mc_simulate_neutral_portfolios(
    const MCNeutralPortfolioConfig *c, double *wealth, double *requested,
    double *funded, std::int8_t *events, double *cost_totals, std::uint8_t *margin_calls
) {
    if (!c || !wealth || !requested || !funded || !events || !cost_totals || !margin_calls ||
        !c->returns || !c->weights || !c->fee_logs || !c->phase_ids ||
        !c->phase_amounts || !c->due_factors || !c->one_times || !c->reviews ||
        c->periods <= 0 || c->paths <= 0 || c->assets <= 0 || c->mode < 0 || c->mode > 2 ||
        (c->mode == 2 && c->rebalance_frequency <= 0)) return 1;
    std::atomic<int> failure{0};
    parallel_paths(c->paths, c->threads, [&](int begin, int end) {
        std::vector<double> holdings(c->assets), allocation(c->assets), fees(c->assets);
        for (int a = 0; a < c->assets; ++a) fees[a] = std::exp(c->fee_logs[a]);
        for (int path = begin; path < end; ++path) {
            double value = c->initial_value, cumulative = c->simple_returns ? 1.0 : 0.0;
            double debt = c->initial_value * (c->leverage - 1.0), costs_total = 0.0;
            bool called = false, cash_flows = c->contribution != 0.0;
            for (int t = 0; t < c->periods; ++t)
                cash_flows = cash_flows || c->phase_ids[t] >= 0 || c->one_times[t] != 0.0;
            for (int a = 0; a < c->assets; ++a)
                holdings[a] = c->initial_value * c->leverage * c->weights[a];
            int phase = -1;
            double annual = 0.0, reference_rate = 0.0;
            double last_wealth = c->initial_value, last_cpi = 1.0;
            for (int t = 0; t < c->periods; ++t) {
                const std::size_t index = static_cast<std::size_t>(t) * c->paths + path;
                const double *returns = c->returns + index * c->assets;
                const double cpi = c->cpi ? c->cpi[index] : 1.0;
                if (c->mode == 0) {
                    double portfolio_return = 0.0;
                    for (int a = 0; a < c->assets; ++a)
                        portfolio_return += (c->simple_returns
                            ? (1.0 + returns[a]) * fees[a] - 1.0
                            : returns[a] + c->fee_logs[a]) * c->weights[a];
                    const double growth = c->simple_returns ? 1.0 + portfolio_return : std::exp(portfolio_return);
                    if (c->simple_returns && !(growth > 0.0)) failure.store(2);
                    if (cash_flows) value = (value + c->contribution) * growth;
                    else {
                        if (c->simple_returns) cumulative *= growth;
                        else cumulative += portfolio_return;
                        value = c->initial_value * (c->simple_returns ? cumulative : std::exp(cumulative));
                    }
                } else {
                    if (c->contribution != 0.0) {
                        if (c->mode == 2) {
                            for (int a = 0; a < c->assets; ++a)
                                holdings[a] += c->contribution * c->leverage * c->weights[a];
                            debt += c->contribution * (c->leverage - 1.0);
                        } else {
                            allocate_contribution(holdings, c->weights, c->contribution, c->contribution_mode, allocation);
                            for (int a = 0; a < c->assets; ++a) holdings[a] += allocation[a];
                        }
                    }
                    for (int a = 0; a < c->assets; ++a) {
                        const double growth = c->simple_returns
                            ? (1.0 + returns[a]) * fees[a] : std::exp(returns[a] + c->fee_logs[a]);
                        if (!std::isfinite(growth) || (c->simple_returns && !(growth > 0.0))) failure.store(2);
                        holdings[a] *= growth;
                    }
                    if (c->mode == 2) debt *= c->financing_paths ? c->financing_paths[index] : c->financing_growth;
                    value = sum_values(holdings) - (c->mode == 2 ? debt : 0.0);
                }
                const double available = std::max(value, 0.0);
                double spending = 0.0;
                std::int8_t event = 0;
                if (c->phase_ids[t] >= 0) {
                    const double base = c->phase_amounts[t];
                    if (phase != c->phase_ids[t]) {
                        phase = c->phase_ids[t]; annual = base * cpi;
                        reference_rate = available > 0.0 ? annual / available : INFINITY;
                        last_wealth = available; last_cpi = cpi;
                    } else if (c->reviews[t] && c->guardrail_policy) {
                        const double real_return = last_wealth > 0.0
                            ? (available / std::max(cpi, 1e-300)) / (last_wealth / std::max(last_cpi, 1e-300)) - 1.0
                            : -2.0;
                        if (!c->skip_inflation_after_loss || real_return >= 0.0)
                            annual = annual * cpi / std::max(last_cpi, 1e-300);
                        const double rate = available > 0.0 ? annual / available : INFINITY;
                        const double before = annual;
                        if (rate > c->upper_guardrail * reference_rate) { annual *= 1.0 - c->adjustment; event = -1; }
                        else if (rate < c->lower_guardrail * reference_rate) { annual *= 1.0 + c->adjustment; event = 1; }
                        annual = std::clamp(annual, base * c->floor * cpi, base * c->ceiling * cpi);
                        if (std::abs(before - annual) <= 1e-8 + 1e-5 * std::abs(annual)) event = 0;
                        last_wealth = available; last_cpi = cpi;
                    } else if (!c->guardrail_policy) annual = base * cpi;
                    spending = annual * c->due_factors[t];
                }
                spending = std::max(spending + c->one_times[t] * cpi, 0.0);
                const double paid = std::min(spending, available);
                requested[index] = spending; funded[index] = paid; events[index] = event;
                if (c->mode == 0) {
                    if (cash_flows) value = std::max(value - paid, 0.0);
                } else {
                    if (c->mode == 2) {
                        const double total = sum_values(holdings);
                        const double fraction = total > 0.0 ? paid / total : 0.0;
                        for (double &holding : holdings) holding -= holding * fraction;
                        if (spending > value) { std::fill(holdings.begin(), holdings.end(), 0.0); debt = 0.0; called = true; }
                        const double assets = sum_values(holdings);
                        value = assets - debt;
                        if (value <= 0.0 || (assets > 0.0 && value / assets < c->maintenance_margin)) {
                            std::fill(holdings.begin(), holdings.end(), 0.0); debt = value = 0.0; called = true;
                        }
                    } else if (paid > 0.0) {
                        const double fraction = paid / std::max(sum_values(holdings), 1e-300);
                        for (double &holding : holdings) holding = std::max(holding - holding * fraction, 0.0);
                        value = sum_values(holdings);
                    }
                    if (c->rebalance_frequency > 0 && (t + 1) % c->rebalance_frequency == 0) {
                        double turnover = 0.0;
                        for (int a = 0; a < c->assets; ++a)
                            turnover += std::abs(value * c->leverage * c->weights[a] - holdings[a]);
                        const double cost = turnover * (c->cost_paths ? c->cost_paths[index] : c->cost_rate);
                        costs_total += cost; value -= cost;
                        if (c->mode == 2 && !called && value <= 0.0) {
                            std::fill(holdings.begin(), holdings.end(), 0.0); debt = value = 0.0; called = true;
                        }
                        if (c->mode != 2 || !called) {
                            for (int a = 0; a < c->assets; ++a) holdings[a] = value * c->leverage * c->weights[a];
                            if (c->mode == 2) debt = value * (c->leverage - 1.0);
                        }
                    }
                }
                wealth[index] = c->mode == 2 ? std::max(value, 0.0) : value;
                if (!std::isfinite(wealth[index])) failure.store(3);
            }
            cost_totals[path] = costs_total; margin_calls[path] = called;
        }
    });
    return failure.load();
}
