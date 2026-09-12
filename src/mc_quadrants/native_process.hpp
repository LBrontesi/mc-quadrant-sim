// Standalone native regime/macro generation, shared with compact simulation.
extern "C" int mc_simulate_regime_macro_paths(
    int periods, int paths, int states, int threads,
    const MCRegimeProcessConfig *regime, const MCMacroProcessConfig *macro,
    std::uint8_t *regimes, double *values, double *shocks
) {
    if (!regime || !regimes || periods <= 0 || paths <= 0 || states <= 0 || states > 256 ||
        !regime->transition_matrix || !regime->start_probabilities ||
        (regime->duration_model && (!regime->duration_hazards || !regime->duration_hazard_lengths)) ||
        (macro && (states != 4 || macro->dimensions < 2 || !values || !shocks ||
            !macro->latest || !macro->var_coefficient || !macro->var_coefficient_std ||
            !macro->state_centers || !macro->state_innovation_cholesky ||
            (!macro->logistic_membership && !macro->emission_coefficients)))) return 1;
    const int dimensions = macro ? macro->dimensions : 0;
    std::vector<double> logits(states * states);
    for (int i = 0; i < states * states; ++i)
        logits[i] = (1.0 - (macro ? macro->transition_weight : 0.0)) * std::log(std::max(regime->transition_matrix[i], 1e-12));
    const auto generator = dimensions == 3 ? generate_joint_regime_macro_path<3>
        : dimensions == 2 ? generate_joint_regime_macro_path<2> : generate_joint_regime_macro_path<0>;
    std::atomic<int> failure{0};
    parallel_paths(paths, threads, [&](int begin, int end) {
        std::vector<std::uint8_t> path_regimes(periods);
        std::vector<double> path_values(periods * dimensions), path_shocks(periods * dimensions);
        std::vector<std::uint64_t> counts(states);
        MacroPathScratch scratch;
        for (int path = begin; path < end; ++path) {
            const int status = macro
                ? generator(*regime, *macro, states, periods, path, path_regimes, path_shocks, path_values, counts, scratch, logits)
                : generate_regime_path(*regime, states, periods, path, path_regimes, counts);
            if (status) { failure.store(status); continue; }
            for (int period = 0; period < periods; ++period) {
                const std::size_t destination = static_cast<std::size_t>(period) * paths + path;
                regimes[destination] = path_regimes[period];
                for (int dimension = 0; dimension < dimensions; ++dimension) {
                    const double value = path_values[period * dimensions + dimension];
                    if (!std::isfinite(value)) failure.store(3);
                    values[destination * dimensions + dimension] = value;
                    shocks[destination * dimensions + dimension] = path_shocks[period * dimensions + dimension];
                }
            }
        }
    });
    return failure.load();
}
