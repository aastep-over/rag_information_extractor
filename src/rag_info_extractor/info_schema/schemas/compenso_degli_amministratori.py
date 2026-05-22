from typing import ClassVar, Literal

from pydantic import BaseModel, Field


class Rimborso(BaseModel):
    """
    Sezione: Rimborso spese amministratori.
    Domande:
    1) "Agli amministratori spetta il rimborso delle spese?"
    2) "Quali spese sono incluse?"
    Chiavi: spetta_rimborso, spese_incluse
    """

    question: ClassVar[str] = (
        "Agli amministratori spetta il rimborso delle spese? Quali spese sono incluse?"
    )
    spetta_rimborso: Literal["Sì", "No", ""] = Field(
        default="",
        description=(
            'Rispondi SOLO usando la RISPOSTA fornita: estrai "Sì" oppure "No"; '
            'se la RISPOSTA non lo dice → "".'
        ),
    )
    spese_incluse: str = Field(
        default="",
        description=(
            "Rispondi SOLO usando la RISPOSTA fornita: elenco sintetico separato da virgole "
            '(es. "viaggio, vitto"); se non indicato nella RISPOSTA → "".'
        ),
    )


class IndennitaAnnuale(BaseModel):
    """
    Sezione: Indennità/compenso annuale deliberata dai soci.
    Domande:
    1) "I soci possono assegnare un compenso agli amministratori?"
    2) "In che misura?"
    Chiavi: spetta_indennita_da_soci, misura_indennita
    """

    question: ClassVar[str] = (
        "I soci possono assegnare un compenso agli amministratori? In che misura?"
    )
    spetta_indennita_da_soci: Literal["Sì", "No", ""] = Field(
        default="",
        description=(
            'Rispondi SOLO usando la RISPOSTA fornita: estrai "Sì" oppure "No"; '
            'se la RISPOSTA non lo dice → "".'
        ),
    )
    misura_indennita: str = Field(
        default="",
        description=(
            "Solo dalla RISPOSTA: misura di compensione; " 'se non presente → "".'
        ),
    )


class IndennitaCessazione(BaseModel):
    """
    Sezione: Indennità in caso di cessazione della carica.
    Domanda:
    1) "I soci possono accantonare una somma a titolo di indennità per gli amministratori in caso di cessazione della carica?"
    Chiavi: spetta_indennita
    """

    question: ClassVar[str] = (
        "I soci possono accantonare una somma a titolo di indennità per gli amministratori in caso di cessazione della carica?"
    )
    spetta_indennita: Literal["Sì", "No", ""] = Field(
        default="",
        description=(
            'Rispondi SOLO usando la RISPOSTA fornita: estrai "Sì" oppure "No"; '
            'se la RISPOSTA non lo dice → "".'
        ),
    )
