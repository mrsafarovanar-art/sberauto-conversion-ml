"""Проверка входных данных; неизвестные категории допустимы."""
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(strict=True, max_length=512)]


class SessionInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    utm_source: Text | None = None
    utm_medium: Text | None = None
    utm_campaign: Text | None = None
    utm_adcontent: Text | None = None
    utm_keyword: Text | None = None
    device_category: Text | None = None
    device_os: Text | None = None
    device_brand: Text | None = None
    device_model: Text | None = None
    device_screen_resolution: Text | None = None
    device_browser: Text | None = None
    geo_country: Text | None = None
    geo_city: Text | None = None

    @model_validator(mode='after')
    def require_information(self):
        values = self.model_dump(exclude={'device_model'}).values()
        if not any(isinstance(v, str) and v.strip() not in ('', '(not set)', 'nan', 'None', '<NA>') for v in values):
            raise ValueError('Нужен хотя бы один непустой признак визита')
        return self


class BatchInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sessions: list[SessionInput] = Field(min_length=1, max_length=1000)
