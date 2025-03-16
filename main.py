"""
This script demonstrates how to convert a Pydantic model to a JSON schema format compatible with OpenAI's structured output 
requirements and use it to make a request to the OpenAI API.

Classes:
    Person(BaseModel): A Pydantic model representing a person with attributes full_name and date_of_birth.
    OpenAISchema(TypedDict): A TypedDict strictly defining the expected format for OpenAI's response_format schema.

Functions:
    pydantic_to_so_openai_schema(pydantic_model: Type[BaseModel]) -> ResponseFormatJSONSchema:

Usage:
    The script loads environment variables, converts the Person Pydantic model to an OpenAI-compatible JSON schema, 
    and makes a request to the OpenAI API using this schema.
"""

from typing import Any, Dict, Type, TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.shared_params.response_format_json_schema import (
    JSONSchema, ResponseFormatJSONSchema)
from pydantic import BaseModel


class Person(BaseModel):
    """
    A class used to represent a Person.

    Attributes
    ----------
    full_name : str
        The full name of the person.
    date_of_birth : str
        The date of birth of the person in YYYY-MM-DD format.
    """
    full_name: str
    date_of_birth: str


class OpenAISchema(TypedDict):
    """Strictly defines the expected format for OpenAI's response_format schema."""

    type: str
    json_schema: Dict[str, Any]


def pydantic_to_so_openai_schema(
    pydantic_model: Type[BaseModel],
) -> ResponseFormatJSONSchema:
    """
    Converts a Pydantic model to a dictionary strictly following OpenAI's structured output JSON schema format.

    Args:
        pydantic_model (Type[BaseModel]): A Pydantic model class.

    Returns:
        OpenAISchema: A dictionary formatted strictly for OpenAI's response_format.
    """
    pydantic_schema = pydantic_model.model_json_schema()
    openai_schema = {
        "type": "object",  # Ensure it's always an object
        "properties": pydantic_schema.get(
            "properties", {}
        ),  # Extract properties safely
        "required": pydantic_schema.get("required", []),  # Handle missing required keys
        "additionalProperties": False,  # Disable extra fields
    }

    openai_json_schema = JSONSchema(
        name=pydantic_schema.get("title", "GeneratedSChema"),
        schema=openai_schema,
        strict=True,
    )

    response_format_json_schema = ResponseFormatJSONSchema(
        type="json_schema", json_schema=openai_json_schema
    )
    return response_format_json_schema


load_dotenv(override=True)


person_schema = pydantic_to_so_openai_schema(Person)

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is Barack Obama basic information?"}
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": '{\n  "full_name": "Barack Hussein Obama II",\n  "date_of_birth": "1961-08-04"\n}',
                }
            ],
        },
    ],
    response_format=person_schema,
    temperature=1,
    max_completion_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)

print(response)
