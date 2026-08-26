"""Experiment configuration.

A flat grid (``grid_config.json``) expands into typed, nested configs that keep three
independent concerns separate:

    DataConfig    -> what defines the dataset (and therefore the on-disk cache key)
    MethodConfig  -> which intervention model is trained and its hyperparameters
    TrainConfig   -> the frozen repeat model, optimisation, and the CV seeds

The intervention is chosen by a single ``model`` field: any scale parameterisation
(``onion``, ``spiral_rope``, ``low_rank-8``, ...) or ``das``. ``ExperimentConfig`` bundles
the three groups and owns the run name + (de)serialisation.

Each family reads only its own fields, so ``ExperimentConfig.validate`` *resets* the other
family's fields to their defaults (see :data:`SCALE_ONLY_FIELDS` / :data:`DAS_ONLY_FIELDS`).
Without that, a swept-but-unread field still lands in the run name, the dataset cache key
and the saved config, so two runs that are identical in everything the method actually
uses look — and score — like different experiments.
"""
from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# Single source of truth for the enumerated choices used across the pipeline.
DATASETS = ("real-real", "source-modified", "modified-source")
STATE_MODES = ("c", "h", "concat")
EDIT_SAMPLERS = ("uniform", "markov")
DAS_MODEL = "das"            # fixed contiguous blocks, one per position
DAS_AUTO_MODEL = "das_auto"  # learned channel -> variable assignment (Csordas-style)
DAS_MODELS = (DAS_MODEL, DAS_AUTO_MODEL)


@dataclass
class DataConfig:
    """Everything that determines the dataset — and thus the cache key.

    ``dataset`` picks the input/target roles:
      * real-real       — minimal pairs of two real words differing at one position
                          (with ``edit_ngram`` > 1: within one n-position window)
      * source-modified — real word in, synthetically edited word as target
      * modified-source — edited word in, original real word as target
    The ``check_*`` flags filter synthetic edits (unused for real-real).

    The intervention always reads from a *random-context* source: a sequence that carries
    the edited window the counterfactual needs and resamples everything else from the
    attested-bigram chain, so no context can leak through the source.
    """

    dataset: str = "source-modified"
    val_ratio: float = 0.05          # real-real ignores this and uses its own holdout split
    max_seq_len: int = 20
    edit_ngram: int = 1              # 1 | 2 | 3 : the edited unit — phoneme, bigram, trigram
    var_ngram: int = 1               # 1 | 2 | 3 : phonemes per DAS variable (sizes the source
                                     # window) — DAS only, reset to 1 for scale methods
    edit_sampler: str = "uniform"    # uniform | markov : is the replacement attested in context?
    min_word_len: int = 3            # drop words shorter than this; raised to var_ngram if lower
    train_all_pos: bool = False      # False -> one random edited position per word
    check_repeat: bool = True        # keep only *generated* sequences the model still repeats
    check_cv_pattern: bool = False   # keep only edits whose consonant/vowel pattern is attested
    check_n_gram: int = 0            # 0 | 2 | 3 : require attested bi-/tri-grams after the edit

    def validate(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        if self.check_n_gram not in (0, 2, 3):
            raise ValueError(f"check_n_gram must be 0, 2, or 3, got {self.check_n_gram}")
        for name in ("edit_ngram", "var_ngram"):
            if getattr(self, name) not in (1, 2, 3):
                raise ValueError(f"{name} must be 1, 2, or 3, got {getattr(self, name)}")
        if self.edit_sampler not in EDIT_SAMPLERS:
            raise ValueError(f"edit_sampler must be one of {EDIT_SAMPLERS}, got {self.edit_sampler!r}")
        # A word shorter than one variable has no addressable variable at all, so it would
        # train on an empty intervention. Raise the floor to match rather than error out;
        # the effective value is what lands in the run name and the cache key.
        self.min_word_len = max(self.min_word_len, self.var_ngram)

    def cache_fields(self) -> dict[str, Any]:
        """The subset of fields that change the dataset content (feeds the cache key).
        Every field of this dataclass affects the produced examples, so all are keyed."""
        return asdict(self)

    def filter_tokens(self) -> list[str]:
        """Compact tokens for the non-default content filters, shared by the run name and
        the dataset cache filename so both are self-describing and never silently collide.
        ``check_repeat`` is on by default, so it only appears when disabled."""
        tokens: list[str] = []
        if self.edit_ngram > 1:
            tokens.append(f"edit{self.edit_ngram}gram")
        if self.var_ngram > 1:  # always named: it changes the DAS variable count too
            tokens.append(f"var{self.var_ngram}gram")
        if self.edit_sampler != "uniform":
            tokens.append("mk")
        if not self.check_repeat:
            tokens.append("norepeat")
        if self.check_cv_pattern:
            tokens.append("cv")
        if self.check_n_gram:
            tokens.append(f"{self.check_n_gram}gram")
        if self.min_word_len > 1:
            tokens.append(f"min{self.min_word_len}")
        if self.train_all_pos:
            tokens.append("allpos")
        return tokens


@dataclass
class MethodConfig:
    """The intervention to train, selected by ``model``.

    ``model`` is either a scale parameterisation (e.g. ``onion``, ``spiral_rope``,
    ``low_rank-8``) or ``das``. Scale-only fields (embedding_init, train_embedding) and
    DAS-only fields (masking, var_size, reg_loss) are reset to their defaults by
    ``ExperimentConfig.validate`` when the other family is selected.
    """

    model: str = "onion"  # onion| expo_decay | spiral_rope | low_rank-n | das | das_auto
    state_mode: str = "concat"   # c | h | concat
    # --- scale only ---
    embedding_init: str = "none"  # none | pretrained | delta_{h,c,state}_{mean,median}
    train_embedding: bool = False
    # --- das only ---
    masking: bool = True
    var_size: int | None = None
    reg_loss: float = 1e-3

    @property
    def is_das(self) -> bool:
        return self.model in DAS_MODELS

    @property
    def is_autoseg(self) -> bool:
        """Learned channel->variable assignment instead of fixed contiguous blocks."""
        return self.model == DAS_AUTO_MODEL

    def validate(self) -> None:
        if self.state_mode not in STATE_MODES:
            raise ValueError(f"state_mode must be one of {STATE_MODES}, got {self.state_mode!r}")
        if not self.model:
            raise ValueError("model must be a scale parameterisation name or 'das'")

    def is_trivial(self) -> bool:
        """A scale model with a random, frozen embedding has no signal to learn from."""
        return not self.is_das and self.embedding_init == "none" and not self.train_embedding


@dataclass
class TrainConfig:
    """Frozen repeat model, optimisation, and the seeds used for cross-validation."""

    model_name: str = "Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1"
    weights_path: str = "resources/weights/1024_75.pth"  # resolved by intervention.paths
    hidden_size: int = 128
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 200
    patience: int = 7
    min_delta: float = 1e-6
    teacher_forcing: bool = False
    seeds: tuple[int, ...] = (42, 43, 44, 45)  # more than one seed -> cross-validation


# Legacy flat-key aliases -> canonical field names (so old grids/results keep working).
_ALIASES = {"condition": "dataset", "dataset_type": "dataset", "scale_param": "model"}

# Fields exactly one intervention family reads, as (group, field name) pairs. The other
# family's runs reset them to their dataclass defaults, so a swept-but-unread value can
# never fork the run name, the dataset cache key, or the saved config.
SCALE_ONLY_FIELDS = (("method", "embedding_init"), ("method", "train_embedding"))
DAS_ONLY_FIELDS = (("data", "var_ngram"), ("method", "masking"),
                   ("method", "var_size"), ("method", "reg_loss"))
# ``das_auto`` learns the channel -> variable assignment, so the fixed-block knobs of
# plain ``das`` are dead there too.
FIXED_BLOCK_FIELDS = (("method", "masking"), ("method", "var_size"))


def _reset_to_defaults(obj: Any, names: list[str]) -> None:
    """Restore ``names`` on a dataclass instance to the values it was declared with."""
    declared = {f.name: f for f in fields(obj)}
    for name in names:
        f = declared[name]
        setattr(obj, name, f.default_factory() if f.default_factory is not MISSING else f.default)


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def drop_ignored(self) -> None:
        """Reset every field the selected family never reads (see the module docstring).

        Runs *before* ``DataConfig.validate`` because ``var_ngram`` raises the
        ``min_word_len`` floor: a scale run must get the floor its effective (reset)
        ``var_ngram`` implies, not the one the swept value would have implied.
        """
        ignored = SCALE_ONLY_FIELDS if self.method.is_das else DAS_ONLY_FIELDS
        if self.method.is_autoseg:
            ignored += FIXED_BLOCK_FIELDS
        groups: dict[str, list[str]] = {}
        for group, name in ignored:
            groups.setdefault(group, []).append(name)
        for group, names in groups.items():
            _reset_to_defaults(getattr(self, group), names)

    def validate(self) -> ExperimentConfig:
        self.method.validate()
        self.drop_ignored()
        self.data.validate()
        # n-gram edits index an n-gram vocabulary, and a scale method learns that table from
        # scratch ('none') — the only initialisation left here. The repeat model's phoneme
        # table ('pretrained') has no n-gram counterpart, and seeding the table from n-gram
        # delta statistics ('delta_*') was tried and did not train, so both are rejected.
        # ``load_ngram_embedding_from_stats`` still builds the delta table for direct use;
        # no config reaches it (see the note in ``models/additive_intervention.py``).
        if self.data.edit_ngram > 1 and not self.method.is_das:
            if self.method.embedding_init== "pretrained" or self.method.embedding_init.startswith("delta_"):
                raise ValueError(
                    f"edit_ngram > 1 on a scale method learns its n-gram embedding from "
                    f"scratch; use embedding_init='none' (got {self.method.embedding_init!r}). "
                    "'pretrained' has no n-gram table, and 'delta_*' n-gram init did not train."
                )
        return self

    # -- naming -------------------------------------------------------------- #
    def run_name(self) -> str:
        """Stable, human-readable identifier for a config (independent of seed)."""
        m = self.method
        parts = [self.data.dataset, *self.data.filter_tokens(), m.model, f"state_{m.state_mode}"]
        if m.is_das:
            if m.masking and not m.is_autoseg:  # autoseg always masks, by construction
                parts.append("mask")
        else:
            if m.embedding_init == "pretrained":
                parts.append("init_model_embed")
            elif m.embedding_init != "none":
                parts.append(f"init_{m.embedding_init}")
            if m.train_embedding:
                parts.append("train_embed")
        if self.train.teacher_forcing:
            parts.append("tf")
        return "-".join(parts)

    # -- (de)serialisation --------------------------------------------------- #
    def to_flat(self) -> dict[str, Any]:
        """Flat dict for saving/summaries, including legacy aliases for older analysis code."""
        flat = {**asdict(self.data), **asdict(self.method), **asdict(self.train)}
        flat["dataset_type"] = self.data.dataset  # alias read by paper_plots / analysis_plots
        flat["scale_param"] = self.method.model   # alias read by paper_plots / analysis_plots
        return flat

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_flat(), f, indent=2)

    @classmethod
    def from_flat(cls, flat: dict[str, Any]) -> ExperimentConfig:
        """Build a nested config from one flat row (e.g. a grid combination).
        """
        flat = dict(flat)
        kind = flat.pop("kind", None)  # legacy: family selector, now folded into `model`
        # `source_mode` was removed; the source is always random-context. Drop it from old
        # grids/configs, but reject the retired `paired` value loudly.
        if flat.pop("source_mode", "random_context") == "paired":
            raise ValueError("source_mode='paired' has been removed; the source is always random-context")
        for alias, canonical in _ALIASES.items():
            if alias in flat and canonical not in flat:
                flat[canonical] = flat.pop(alias)
            else:
                flat.pop(alias, None)
        if kind == DAS_MODEL:
            flat["model"] = DAS_MODEL
        if "seed" in flat and "seeds" not in flat:  # singular convenience alias
            flat["seeds"] = [flat.pop("seed")]

        groups = {"data": DataConfig, "method": MethodConfig, "train": TrainConfig}
        field_owner = {f.name: name for name, klass in groups.items() for f in fields(klass)}
        unknown = set(flat) - set(field_owner)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")

        buckets: dict[str, dict[str, Any]] = {name: {} for name in groups}
        for key, value in flat.items():
            buckets[field_owner[key]][key] = value
        if "seeds" in buckets["train"]:
            buckets["train"]["seeds"] = tuple(buckets["train"]["seeds"])

        return cls(
            data=DataConfig(**buckets["data"]),
            method=MethodConfig(**buckets["method"]),
            train=TrainConfig(**buckets["train"]),
        ).validate()
