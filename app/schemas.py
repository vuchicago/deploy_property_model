from pydantic import BaseModel, HttpUrl
from typing import Sequence


class Debit(BaseModel):
    Description_all: str


class RecipeSearchResults(BaseModel):
    results: Sequence[Recipe]


class RecipeCreate(BaseModel):
    label: str
    source: str
    url: HttpUrl
    submitter_id: int