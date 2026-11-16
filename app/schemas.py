from pydantic import BaseModel, Field, field_validator


class TextInput(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be empty or contain only whitespace.")
        return value


class PredictionResponse(BaseModel):
    prediction: str
    score: float


class HealthResponse(BaseModel):
    status: str