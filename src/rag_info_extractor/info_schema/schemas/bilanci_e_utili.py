from typing import ClassVar, Literal

from pydantic import BaseModel, Field


class PercentualeRiservaLegale(BaseModel):
    """
    Sezione: Accantonamento utili a riserva legale.
    Domanda:
    1) "Quale percentuale degli utili deve essere accantonata da destinare alla riserva legale?"
    Chiavi: percentuale_utili
    """

    question: ClassVar[str] = (
        "Quale percentuale degli utili deve essere accantonata da destinare alla riserva legale?"
    )
    percentuale_utili: str = Field(
        default="",
        description=(
            "Numero che rappresenta Percentuale di utili da accantonare. Restituisci Solo il numero esempio: 33% → 33 or trentatre per cento → trentatre"
            "Se non specificato nella RISPOSTA → stringa vuota."
        ),
    )


class CapitaleSociale(BaseModel):
    """
    Sezione: Capitale Sociale.
    Domanda:
    1) "Qual è l’ammontare del capitale sociale della società?"
    Chiavi: capitale_sociale_euro
    """

    question: ClassVar[str] = "Qual è l’ammontare del capitale sociale della società?"
    capitale_sociale_euro: str = Field(
        default="",
        description=(
            "Valore numerico in euro del capitale sociale. "
            "Se non specificato nella RISPOSTA → stringa vuota."
        ),
    )


class TermineApprovazioneBilancio(BaseModel):
    """
    Sezione: Termine approvazione bilancio.
    Domanda:
    1) "Entro quanti giorni dalla chiusura dell’esercizio deve essere approvato il bilancio?"
    2) "Qual è il termine prorogato in presenza di particolari esigenze?"
    Chiavi: termine_ordinario_giorni, termine_prorogato_giorni
    """

    question: ClassVar[str] = (
        "Entro quanti giorni dalla chiusura dell’esercizio deve essere approvato il bilancio normalmente? Qual è il termine prorogato in presenza di particolari esigenze?"
    )
    termine_ordinario_giorni: str = Field(
        default="",
        description=(
            "Tempistiche per l'approvazione del bilancio in condizioni ordinarie;"
            "Accettare anche risposte del tipo 'come previsto dalla legge' o 'secondo termini di legge';"
            "Non convertire/normalizzare (non trasformare “duecento” in “200”);"
            "Se non specificato → stringa vuota."
        ),
        # json_schema_extra={"word_to_number": True}, # need to parse through word_to_number once returned by llm
    )
    termine_prorogato_giorni: str = Field(
        default="",
        description=(
            "Tempistiche prorogato per l'approvazione del bilancio in presenza di particolari esigenze;"
            "Non convertire/normalizzare (non trasformare “duecento” in “200”);"
            "Se non specificato → stringa vuota."
        ),
        # json_schema_extra={"word_to_number": True}, # need to parse through word_to_number once returned by llm
    )

    # Store which functions to apply post-process(after generation) and to which variables of the class
    post_process_func_var: ClassVar[dict[str, list]] = {
        "formatted_word_to_number": [
            "termine_ordinario_giorni",
            "termine_prorogato_giorni",
        ],
    }


class DataChiusuraEsercizio(BaseModel):
    """
    Sezione: Data di chiusura esercizio sociale.
    Domanda:
    1) "Qual è la data di chiusura dell’esercizio sociale?"
    Chiavi: data_chiusura_esercizio
    """

    question: ClassVar[str] = "Qual è la data di chiusura dell’esercizio sociale?"
    data_chiusura_esercizio: str = Field(
        default="",
        description=(
            "Data di chiusura dell'esercizio come appare nel documento (es. '30 dicembre'). "
            "Se non specificato → stringa vuota."
        ),
    )


class UtiliResidui(BaseModel):
    """
    Sezione: Destinazione utili residui.
    Domanda:
    1) "Cosa si fanno degli utili residui secondo lo statuto?"
    Chiavi: utili_residui
    """

    question: ClassVar[str] = "Cosa si fanno degli utili residui secondo lo statuto?"
    utili_residui: str = Field(
        default="",
        description=(
            "Sintesi della destinazione degli utili residui. "
            "Se non specificato → stringa vuota."
        ),
    )
