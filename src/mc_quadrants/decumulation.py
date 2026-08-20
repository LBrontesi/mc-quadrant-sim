from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np


DECUMULATION_MODES = {"manual", "safe_rate"}
DECUMULATION_POLICIES = {"fixed", "guyton_klinger"}
DECUMULATION_TARGETS = {"survival", "preserve_initial", "minimum_bequest"}
FREQUENCY_MONTHS = {"monthly": 1, "quarterly": 3, "annual": 12}


def _finite_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = f" at least {minimum:g}" if minimum is not None else ""
        raise ValueError(f"{name} must be a finite number{qualifier}.")
    return result


def _integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    numeric = _finite_float(value, name)
    integer = int(numeric)
    if not np.isclose(numeric, integer):
        raise ValueError(f"{name} must be an integer.")
    if not minimum <= integer <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return integer


@dataclass(frozen=True)
class WithdrawalPhase:
    """One inclusive, non-overlapping recurring spending interval."""

    start_month: int
    end_month: int
    frequency: str = "monthly"
    annual_real_amount: float = 0.0
    spending_multiplier: float = 1.0

    @property
    def frequency_months(self) -> int:
        return FREQUENCY_MONTHS[self.frequency]

    def due(self, month: int) -> bool:
        return (
            self.start_month <= month <= self.end_month
            and (month - self.start_month) % self.frequency_months == 0
        )

    def annual_amount(self, *, safe_rate: float, initial_value: float, mode: str) -> float:
        if mode == "safe_rate":
            return initial_value * safe_rate * self.spending_multiplier
        return self.annual_real_amount

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_month": self.start_month,
            "end_month": self.end_month,
            "frequency": self.frequency,
            "annual_real_amount": self.annual_real_amount,
            "spending_multiplier": self.spending_multiplier,
        }


@dataclass(frozen=True)
class OneTimeExpense:
    month: int
    real_amount: float

    def to_dict(self) -> dict[str, Any]:
        return {"month": self.month, "real_amount": self.real_amount}


@dataclass(frozen=True)
class GuardrailSettings:
    policy: str = "fixed"
    review_months: int = 12
    upper_guardrail: float = 1.20
    lower_guardrail: float = 0.80
    adjustment: float = 0.10
    floor: float = 0.70
    ceiling: float = 1.30
    skip_inflation_after_negative_real_return: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.policy,
            "review_months": self.review_months,
            "upper_guardrail": self.upper_guardrail,
            "lower_guardrail": self.lower_guardrail,
            "adjustment": self.adjustment,
            "floor": self.floor,
            "ceiling": self.ceiling,
            "skip_inflation_after_negative_real_return": (
                self.skip_inflation_after_negative_real_return
            ),
        }


@dataclass(frozen=True)
class SafeRateSettings:
    objective: str = "survival"
    target_probability: float = 0.90
    minimum_bequest: float = 0.0
    maximum_rate: float = 0.25
    precision: float = 0.0005

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "target_probability": self.target_probability,
            "minimum_bequest": self.minimum_bequest,
            "maximum_rate": self.maximum_rate,
            "precision": self.precision,
        }


@dataclass(frozen=True)
class DecumulationPlan:
    """Normalized advanced decumulation contract used by every backend."""

    enabled: bool = False
    mode: str = "manual"
    phases: tuple[WithdrawalPhase, ...] = ()
    one_time_expenses: tuple[OneTimeExpense, ...] = ()
    guardrails: GuardrailSettings = field(default_factory=GuardrailSettings)
    safe_rate: SafeRateSettings = field(default_factory=SafeRateSettings)
    annual_inflation_fallback: float = 0.0
    legacy_nominal: bool = False

    @property
    def policy(self) -> str:
        return self.guardrails.policy

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.phases or self.one_time_expenses)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "phases": [phase.to_dict() for phase in self.phases],
            "one_time_expenses": [event.to_dict() for event in self.one_time_expenses],
            "policy": self.guardrails.to_dict(),
            "safe_rate": self.safe_rate.to_dict(),
            "annual_inflation_fallback": self.annual_inflation_fallback,
            "legacy_nominal": self.legacy_nominal,
        }


def _parse_policy(raw: Any) -> GuardrailSettings:
    values: Mapping[str, Any]
    if isinstance(raw, Mapping):
        values = raw
        policy = str(values.get("type", values.get("policy", "fixed"))).strip().lower()
    else:
        values = {}
        policy = str(raw or "fixed").strip().lower()
    if policy not in DECUMULATION_POLICIES:
        raise ValueError("decumulation.policy must be fixed or guyton_klinger.")
    review_months = _integer(
        values.get("review_months", 12),
        "decumulation.policy.review_months",
        minimum=1,
        maximum=120,
    )
    upper = _finite_float(
        values.get("upper_guardrail", 1.20),
        "decumulation.policy.upper_guardrail",
        minimum=0.0,
    )
    lower = _finite_float(
        values.get("lower_guardrail", 0.80),
        "decumulation.policy.lower_guardrail",
        minimum=0.0,
    )
    adjustment = _finite_float(
        values.get("adjustment", 0.10),
        "decumulation.policy.adjustment",
        minimum=0.0,
    )
    floor = _finite_float(values.get("floor", 0.70), "decumulation.policy.floor", minimum=0.0)
    ceiling = _finite_float(
        values.get("ceiling", 1.30),
        "decumulation.policy.ceiling",
        minimum=0.0,
    )
    if not lower < upper:
        raise ValueError("decumulation policy lower_guardrail must be below upper_guardrail.")
    if adjustment >= 1:
        raise ValueError("decumulation policy adjustment must be below 1.")
    if not floor <= 1.0 <= ceiling:
        raise ValueError("decumulation policy floor and ceiling must contain 1.0.")
    return GuardrailSettings(
        policy=policy,
        review_months=review_months,
        upper_guardrail=upper,
        lower_guardrail=lower,
        adjustment=adjustment,
        floor=floor,
        ceiling=ceiling,
        skip_inflation_after_negative_real_return=bool(
            values.get("skip_inflation_after_negative_real_return", True)
        ),
    )


def _parse_safe_rate(raw: Any, top_level: Mapping[str, Any]) -> SafeRateSettings:
    values = raw if isinstance(raw, Mapping) else {}
    objective = str(
        values.get("objective", top_level.get("objective", "survival"))
    ).strip().lower()
    aliases = {
        "capital_preservation": "preserve_initial",
        "preservation": "preserve_initial",
        "bequest": "minimum_bequest",
    }
    objective = aliases.get(objective, objective)
    if objective not in DECUMULATION_TARGETS:
        raise ValueError(
            "decumulation safe-rate objective must be survival, preserve_initial, or minimum_bequest."
        )
    probability = _finite_float(
        values.get("target_probability", top_level.get("target_probability", 0.90)),
        "decumulation.safe_rate.target_probability",
    )
    if probability > 1 and probability <= 100:
        probability /= 100.0
    if not 0 < probability < 1:
        raise ValueError("decumulation safe-rate target_probability must be between 0 and 1.")
    minimum_bequest = _finite_float(
        values.get("minimum_bequest", top_level.get("minimum_bequest", 0.0)),
        "decumulation.safe_rate.minimum_bequest",
        minimum=0.0,
    )
    maximum_rate = _finite_float(
        values.get("maximum_rate", 0.25),
        "decumulation.safe_rate.maximum_rate",
        minimum=0.0,
    )
    precision = _finite_float(
        values.get("precision", 0.0005),
        "decumulation.safe_rate.precision",
        minimum=1e-8,
    )
    return SafeRateSettings(
        objective=objective,
        target_probability=probability,
        minimum_bequest=minimum_bequest,
        maximum_rate=maximum_rate,
        precision=precision,
    )


def normalize_decumulation(
    raw: Mapping[str, Any] | DecumulationPlan | None,
    *,
    periods: int,
    legacy_withdrawal: float = 0.0,
    legacy_start_period: int = 1,
    annual_inflation_fallback: float = 0.0,
) -> DecumulationPlan:
    """Validate the API object and migrate legacy scalar withdrawals."""

    periods = _integer(periods, "periods", minimum=1, maximum=360)
    if isinstance(raw, DecumulationPlan):
        return raw
    if raw is None:
        legacy = _finite_float(legacy_withdrawal, "withdrawal", minimum=0.0)
        start = _integer(
            legacy_start_period,
            "withdrawal_start_period",
            minimum=1,
            maximum=periods,
        )
        if np.isclose(legacy, 0.0):
            return DecumulationPlan(
                annual_inflation_fallback=float(annual_inflation_fallback),
                legacy_nominal=True,
            )
        return DecumulationPlan(
            enabled=True,
            phases=(
                WithdrawalPhase(
                    start_month=start,
                    end_month=periods,
                    frequency="monthly",
                    annual_real_amount=legacy * 12.0,
                ),
            ),
            annual_inflation_fallback=float(annual_inflation_fallback),
            legacy_nominal=True,
        )
    if not isinstance(raw, Mapping):
        raise ValueError("decumulation must be an object.")

    enabled = bool(raw.get("enabled", True))
    mode = str(raw.get("mode", "manual")).strip().lower()
    if mode not in DECUMULATION_MODES:
        raise ValueError("decumulation.mode must be manual or safe_rate.")
    fallback = _finite_float(
        raw.get("annual_inflation_fallback", annual_inflation_fallback),
        "decumulation.annual_inflation_fallback",
    )
    if fallback <= -1:
        raise ValueError("decumulation annual_inflation_fallback must be above -100%.")

    raw_phases = raw.get("phases", [])
    if not isinstance(raw_phases, Sequence) or isinstance(raw_phases, (str, bytes)):
        raise ValueError("decumulation.phases must be an array.")
    phases: list[WithdrawalPhase] = []
    for index, item in enumerate(raw_phases):
        if not isinstance(item, Mapping):
            raise ValueError(f"decumulation.phases[{index}] must be an object.")
        start = _integer(
            item.get("start_month", item.get("start_period", 1)),
            f"decumulation.phases[{index}].start_month",
            minimum=1,
            maximum=periods,
        )
        end = _integer(
            item.get("end_month", item.get("end_period", periods)),
            f"decumulation.phases[{index}].end_month",
            minimum=1,
            maximum=periods,
        )
        if end < start:
            raise ValueError(f"decumulation.phases[{index}] must end on or after it starts.")
        frequency = str(item.get("frequency", "monthly")).strip().lower()
        if frequency not in FREQUENCY_MONTHS:
            raise ValueError("decumulation phase frequency must be monthly, quarterly, or annual.")
        amount = _finite_float(
            item.get("annual_real_amount", item.get("amount", 0.0)),
            f"decumulation.phases[{index}].annual_real_amount",
            minimum=0.0,
        )
        multiplier = _finite_float(
            item.get("spending_multiplier", item.get("multiplier", 1.0)),
            f"decumulation.phases[{index}].spending_multiplier",
            minimum=0.0,
        )
        if mode == "manual" and np.isclose(amount, 0.0):
            raise ValueError("Manual decumulation phases require a positive annual_real_amount.")
        if mode == "safe_rate" and np.isclose(multiplier, 0.0):
            raise ValueError("Safe-rate decumulation phases require a positive spending_multiplier.")
        phases.append(
            WithdrawalPhase(start, end, frequency, amount, multiplier)
        )
    phases.sort(key=lambda phase: (phase.start_month, phase.end_month))
    for previous, current in zip(phases, phases[1:]):
        if current.start_month <= previous.end_month:
            raise ValueError("Recurring decumulation phases cannot overlap.")

    raw_events = raw.get("one_time_expenses", raw.get("one_time", []))
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
        raise ValueError("decumulation.one_time_expenses must be an array.")
    events: dict[int, float] = {}
    for index, item in enumerate(raw_events):
        if not isinstance(item, Mapping):
            raise ValueError(f"decumulation.one_time_expenses[{index}] must be an object.")
        month = _integer(
            item.get("month", item.get("period", 1)),
            f"decumulation.one_time_expenses[{index}].month",
            minimum=1,
            maximum=periods,
        )
        amount = _finite_float(
            item.get("real_amount", item.get("amount", 0.0)),
            f"decumulation.one_time_expenses[{index}].real_amount",
            minimum=0.0,
        )
        events[month] = events.get(month, 0.0) + amount

    policy_raw = raw.get("policy", "fixed")
    guardrails = _parse_policy(policy_raw)
    safe_rate = _parse_safe_rate(raw.get("safe_rate"), raw)
    if enabled and not phases and not events:
        raise ValueError("Enabled decumulation requires at least one phase or one-time expense.")
    return DecumulationPlan(
        enabled=enabled,
        mode=mode,
        phases=tuple(phases),
        one_time_expenses=tuple(
            OneTimeExpense(month, amount) for month, amount in sorted(events.items())
        ),
        guardrails=guardrails,
        safe_rate=safe_rate,
        annual_inflation_fallback=fallback,
        legacy_nominal=False,
    )


def inflation_index(
    periods: int,
    paths: int,
    *,
    annual_inflation: float = 0.0,
    inflation_paths: np.ndarray | None = None,
) -> np.ndarray:
    """Return end-of-month nominal units per unit of initial purchasing power."""

    if inflation_paths is None:
        rates = np.full((periods, paths), float(annual_inflation), dtype=float)
    else:
        rates = np.asarray(inflation_paths, dtype=float)
        if rates.shape != (periods, paths):
            raise ValueError("inflation_paths must have shape (periods, paths).")
    if not np.isfinite(rates).all() or (rates <= -1.0).any():
        raise ValueError("inflation paths must contain finite annual rates above -100%.")
    return np.cumprod(np.power(1.0 + rates, 1.0 / 12.0), axis=0)


class SpendingController:
    """Path-aware fixed or Guyton-Klinger monthly spending request engine."""

    def __init__(
        self,
        plan: DecumulationPlan,
        *,
        paths: int,
        initial_value: float,
        cpi: np.ndarray,
        safe_rate: float = 0.0,
    ) -> None:
        self.plan = plan
        self.paths = int(paths)
        self.initial_value = float(initial_value)
        self.cpi = np.asarray(cpi, dtype=float)
        self.safe_rate = float(safe_rate)
        if plan.mode == "safe_rate" and self.safe_rate < 0:
            raise ValueError("safe_rate must be non-negative.")
        self.current_phase = -1
        self.current_annual_nominal = np.zeros(paths, dtype=float)
        self.reference_rate = np.zeros(paths, dtype=float)
        self.last_review_wealth = np.full(paths, initial_value, dtype=float)
        self.last_review_cpi = np.ones(paths, dtype=float)

    def _phase_index(self, month: int) -> int:
        for index, phase in enumerate(self.plan.phases):
            if phase.start_month <= month <= phase.end_month:
                return index
        return -1

    def request(self, month: int, wealth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wealth = np.asarray(wealth, dtype=float)
        if wealth.shape != (self.paths,):
            raise ValueError("wealth must have one value for every path.")
        requested = np.zeros(self.paths, dtype=float)
        events = np.zeros(self.paths, dtype=np.int8)
        if not self.plan.active:
            return requested, events
        cpi = np.ones(self.paths, dtype=float) if self.plan.legacy_nominal else self.cpi[month - 1]
        phase_index = self._phase_index(month)
        if phase_index >= 0:
            phase = self.plan.phases[phase_index]
            base_real = phase.annual_amount(
                safe_rate=self.safe_rate,
                initial_value=self.initial_value,
                mode=self.plan.mode,
            )
            reset = phase_index != self.current_phase
            review = reset or (month - phase.start_month) % self.plan.guardrails.review_months == 0
            if reset:
                self.current_phase = phase_index
                self.current_annual_nominal = base_real * cpi
                self.reference_rate = np.divide(
                    self.current_annual_nominal,
                    wealth,
                    out=np.full(self.paths, np.inf),
                    where=wealth > 0,
                )
                self.last_review_wealth = wealth.copy()
                self.last_review_cpi = cpi.copy()
            elif review and self.plan.policy == "guyton_klinger":
                real_return = np.divide(
                    wealth / np.maximum(cpi, 1e-300),
                    self.last_review_wealth / np.maximum(self.last_review_cpi, 1e-300),
                    out=np.full(self.paths, -1.0),
                    where=self.last_review_wealth > 0,
                ) - 1.0
                index_spending = np.ones(self.paths, dtype=bool)
                if self.plan.guardrails.skip_inflation_after_negative_real_return:
                    index_spending = real_return >= 0.0
                indexed = self.current_annual_nominal * cpi / np.maximum(
                    self.last_review_cpi, 1e-300
                )
                self.current_annual_nominal = np.where(
                    index_spending, indexed, self.current_annual_nominal
                )
                current_rate = np.divide(
                    self.current_annual_nominal,
                    wealth,
                    out=np.full(self.paths, np.inf),
                    where=wealth > 0,
                )
                cut = current_rate > self.plan.guardrails.upper_guardrail * self.reference_rate
                raise_spending = (
                    current_rate < self.plan.guardrails.lower_guardrail * self.reference_rate
                )
                before = self.current_annual_nominal.copy()
                self.current_annual_nominal[cut] *= 1.0 - self.plan.guardrails.adjustment
                self.current_annual_nominal[raise_spending] *= 1.0 + self.plan.guardrails.adjustment
                floor = base_real * self.plan.guardrails.floor * cpi
                ceiling = base_real * self.plan.guardrails.ceiling * cpi
                self.current_annual_nominal = np.clip(
                    self.current_annual_nominal, floor, ceiling
                )
                changed = ~np.isclose(before, self.current_annual_nominal)
                events[cut & changed] = -1
                events[raise_spending & changed] = 1
                self.last_review_wealth = wealth.copy()
                self.last_review_cpi = cpi.copy()
            elif self.plan.policy == "fixed":
                self.current_annual_nominal = base_real * cpi
            if phase.due(month):
                requested += self.current_annual_nominal * phase.frequency_months / 12.0

        for expense in self.plan.one_time_expenses:
            if expense.month == month:
                requested += expense.real_amount * cpi
        return np.maximum(requested, 0.0), events


def funded_amount(requested: np.ndarray, available: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(requested, 0.0), np.maximum(available, 0.0))


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval; 95% is the supported reporting default."""

    if trials <= 0:
        raise ValueError("trials must be positive.")
    if not np.isclose(confidence, 0.95):
        raise ValueError("Only the 95% Wilson interval is currently supported.")
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    radius = z / denominator * sqrt(
        probability * (1.0 - probability) / trials + z * z / (4.0 * trials * trials)
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def success_mask(
    wealth: np.ndarray,
    requested: np.ndarray,
    funded: np.ndarray,
    *,
    objective: str,
    initial_value: float,
    minimum_bequest: float = 0.0,
    terminal_cpi: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate fully funded spending and the selected terminal objective."""

    values = np.asarray(wealth, dtype=float)
    fully_funded = np.all(
        np.asarray(funded, dtype=float) + 1e-8 >= np.asarray(requested, dtype=float),
        axis=0,
    )
    terminal = values[-1]
    if terminal_cpi is not None:
        terminal = terminal / np.maximum(np.asarray(terminal_cpi, dtype=float), 1e-300)
    if objective == "survival":
        return fully_funded
    if objective == "preserve_initial":
        return fully_funded & (terminal >= float(initial_value))
    if objective == "minimum_bequest":
        return fully_funded & (terminal >= float(minimum_bequest))
    raise ValueError(f"Unknown safe-rate objective '{objective}'.")
