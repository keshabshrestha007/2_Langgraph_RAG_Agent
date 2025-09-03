from pydantic import BaseModel,Field
from typing import Literal


class ValidateQuestion(BaseModel):
    """
    Validates whether the input question is related to specific topic.
    
    Returns:
        'Yes'-> if question is related to the topic
        'No'-> if question is not related to the topic
    """
    score:Literal["Yes","No"] = Field(...,
                                      description="Is question about the specified topic? If yes -> 'Yes' and if no ->'No' " 
                                      )
# Document relevance validation model
class GradeDocument(BaseModel):
    """verifies the retrieved documents is relevant to user's question."""
    score:Literal["Yes","No"] = Field(
        ...,
        description="Is document relevant to user's question? If yes ->'Yes' ,if no ->'No'"
    )