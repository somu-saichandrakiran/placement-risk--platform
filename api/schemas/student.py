# api/schemas/student.py
from pydantic import BaseModel, Field
from typing import Optional

class StudentInput(BaseModel):
    tenth_percent:      float = Field(ge=0.0,  le=100.0)
    twelfth_percent:    float = Field(ge=0.0,  le=100.0)
    cgpa:               float = Field(ge=0.0,  le=100.0)
    employability_score:float = Field(ge=0.0,  le=100.0)
    mba_percent:        float = Field(ge=0.0,  le=100.0)
    internship_count:   int   = Field(ge=0,    le=10)
    project_count:      int   = Field(ge=0,    le=20)
    hackathon_count:    int   = Field(ge=0,    le=10)
    active_backlogs:    int   = Field(ge=0,    le=20)
    work_exp_flag:      int   = Field(ge=0,    le=1)
    specialization_12th:str
    degree_type:        str
    specialization:     str
    college_tier:       Optional[str] = "T2"

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenth_percent":       78.5,
                "twelfth_percent":     72.0,
                "cgpa":                67.5,
                "employability_score": 74.0,
                "mba_percent":         62.0,
                "internship_count":    1,
                "project_count":       3,
                "hackathon_count":     1,
                "active_backlogs":     0,
                "work_exp_flag":       0,
                "specialization_12th": "Science",
                "degree_type":         "Sci&Tech",
                "specialization":      "Mkt&Fin",
                "college_tier":        "T2"
            }
        }
    }