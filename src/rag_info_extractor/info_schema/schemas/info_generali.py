from pydantic import BaseModel, Field
from typing import Literal, ClassVar

class Durata(BaseModel):
    """
    Sezione: Durata della società.
    Domanda:
    1) "Qual è la durata della società (fino a quale data)?"
    Chiavi: durata
    """
    question: ClassVar[str] = "Qual è la durata della società (fino a quale data)?"
    durata_dell_azienda: str = Field(
        default="",
        description=(
            "Data di durata della società come appare nel documento (es. '30 dicembre 2050')"
            "Se non specificato → stringa vuota."
        )
    )