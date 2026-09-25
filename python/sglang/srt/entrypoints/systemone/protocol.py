"""Pydantic models for the System One decision API.

Follows the published System One OpenAPI 0.2.0 request and response shapes:
a state, a map of noul, choice, and score questions keyed by caller ids, and
one answer per question id.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from sglang.srt.entrypoints.openai.protocol import (
    DecisionText,
    RequiredDecisionText,
    check_option_names,
    is_blank_decision_text,
)

# The documented maximum. Options beyond 26 get two-letter labels, checked per request.
MAX_CHOICE_OPTIONS = 255
# Levels are labeled 0 to 9, so at most 10.
MAX_SCORE_LEVELS = 10


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class _Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The x_fingerprint of the answers it was fitted on. When sent, a request
    # whose reads or model differ is refused instead of calibrated wrongly.
    fitted_on: Optional[str] = None


class TemperatureCalibration(_Calibration):
    """Divides every option's log probability, for any question type."""

    type: Literal["temperature"]
    temperature: FiniteFloat = Field(gt=0)


class PlattCalibration(_Calibration):
    """Rescales the log odds of yes, for noul questions: P(yes) = sigmoid(a * z + b)."""

    type: Literal["platt"]
    a: FiniteFloat
    b: FiniteFloat


class VectorCalibration(_Calibration):
    """A scale and bias per option on its log probability, for choice and score."""

    type: Literal["vector"]
    scale: List[FiniteFloat] = Field(min_length=1)
    bias: List[FiniteFloat] = Field(min_length=1)


SystemOneCalibration = Annotated[
    Union[TemperatureCalibration, PlattCalibration, VectorCalibration],
    Field(discriminator="type"),
]


class _Question(BaseModel):
    # Misspelled keys inside a question would otherwise answer a different question.
    model_config = ConfigDict(extra="forbid")

    instructions: Optional[DecisionText] = None
    # SGLang extension: a calibration the client fitted for this question,
    # applied after its reads are combined.
    x_calibration: Optional[SystemOneCalibration] = None

    def _check_calibration(self, allowed: tuple, options: int) -> None:
        calibration = self.x_calibration
        if calibration is None:
            return
        if calibration.type not in allowed:
            raise ValueError(
                f"a {self.type} question takes a {' or '.join(allowed)} "
                f"calibration, not {calibration.type}"
            )
        if isinstance(calibration, VectorCalibration) and not (
            len(calibration.scale) == len(calibration.bias) == options
        ):
            raise ValueError(
                f"vector calibration needs a scale and bias for each of the "
                f"{options} options"
            )


class SystemOneNoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: Optional[DecisionText] = None
    false: Optional[DecisionText] = None


class SystemOneNoulQuestion(_Question):
    type: Literal["noul"]
    criteria: Optional[SystemOneNoulCriteria] = None

    @model_validator(mode="after")
    def _asks_something(self):
        criteria = self.criteria or SystemOneNoulCriteria()
        if all(
            is_blank_decision_text(value)
            for value in (self.instructions, criteria.true, criteria.false)
        ):
            raise ValueError(
                "a noul question needs instructions or a true or false "
                "description to decide on"
            )
        self._check_calibration(("temperature", "platt"), options=2)
        return self


class SystemOneChoiceQuestion(_Question):
    type: Literal["choice"]
    criteria: Dict[str, Optional[DecisionText]] = Field(
        min_length=1, max_length=MAX_CHOICE_OPTIONS
    )

    @field_validator("criteria")
    @classmethod
    def _option_names_distinct(cls, criteria):
        check_option_names(criteria)
        return criteria

    @model_validator(mode="after")
    def _calibration_fits(self):
        self._check_calibration(("temperature", "vector"), options=len(self.criteria))
        return self


class SystemOneScoreQuestion(_Question):
    type: Literal["score"]
    criteria: List[RequiredDecisionText] = Field(
        min_length=1, max_length=MAX_SCORE_LEVELS
    )

    @model_validator(mode="after")
    def _calibration_fits(self):
        self._check_calibration(("temperature", "vector"), options=len(self.criteria))
        return self


SystemOneQuestion = Annotated[
    Union[SystemOneNoulQuestion, SystemOneChoiceQuestion, SystemOneScoreQuestion],
    Field(discriminator="type"),
]


class SystemOneReadSetup(BaseModel):
    """How to read each question, as the server's reads config names the fields."""

    model_config = ConfigDict(extra="forbid")

    choice_rotations: int = Field(ge=1)
    choice_name_variants: bool
    noul_orders: int = Field(ge=1, le=2)
    noul_case_variants: bool


class SystemOneRequest(BaseModel):
    # Unknown top-level fields are ignored, as the published schema allows.
    state: DecisionText
    model: str
    questions: Dict[str, SystemOneQuestion] = Field(min_length=1)
    # SGLang extension, for chat templates whose reasoning toggle needs a kwarg.
    chat_template_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # SGLang extensions: how to read every question, the server's default reads
    # when not sent, and whether answers include the label logprobs of every read.
    x_read_setup: Optional[SystemOneReadSetup] = None
    x_return_reads: bool = False

    # /v1/decisions fields, which would change the answers if honored or ignored.
    # Declared only to refuse them by name, and hidden from the schema.
    temperature: SkipJsonSchema[Any] = None
    prompt_format_version: SkipJsonSchema[Any] = None
    return_prompt_token_ids: SkipJsonSchema[Any] = None

    @field_validator("temperature", "prompt_format_version", "return_prompt_token_ids")
    @classmethod
    def _refuse_decisions_fields(cls, value, info: ValidationInfo):
        if value is not None:
            raise ValueError(
                f"{info.field_name} is not part of this API, use /v1/decisions for it"
            )
        return value


class SystemOneRead(BaseModel):
    """One scored prompt of a question, an SGLang extension."""

    # Option names in the order the prompt shows them, with their labels.
    order: List[str]
    labels: List[str]
    # Per option name, the texts scored for it and their full-vocabulary logprobs.
    texts: Dict[str, List[str]]
    logprobs: Dict[str, List[float]]


# Answer fields are declared per type so the published fields come first. The
# x_ fields are SGLang extensions: x_label_mass is the full-vocabulary
# probability of the tokens scored in the read that shows options in request
# order, x_read_probabilities the option probabilities of the combined reads
# before any calibration, x_calibration the calibration applied, x_fingerprint
# what a calibration fitted on this answer must be sent with, and x_reads every
# read when the request sets x_return_reads.
CalibrationApplied = Literal["none", "temperature", "platt", "vector"]


class SystemOneNoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float
    x_label_mass: float
    x_read_probabilities: Dict[str, float]
    x_calibration: CalibrationApplied
    x_fingerprint: str
    x_reads: Optional[List[SystemOneRead]] = None


class SystemOneChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: Dict[str, float]
    x_label_mass: float
    x_read_probabilities: Dict[str, float]
    x_calibration: CalibrationApplied
    x_fingerprint: str
    x_reads: Optional[List[SystemOneRead]] = None


class SystemOneScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: Dict[str, Any]
    probabilities: Dict[str, float]
    x_label_mass: float
    x_read_probabilities: Dict[str, float]
    x_calibration: CalibrationApplied
    x_fingerprint: str
    x_reads: Optional[List[SystemOneRead]] = None


class SystemOneUsage(BaseModel):
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: Dict[
        str, Union[SystemOneNoulAnswer, SystemOneChoiceAnswer, SystemOneScoreAnswer]
    ]
    usage: SystemOneUsage
