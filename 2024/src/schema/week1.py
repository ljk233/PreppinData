from typing import NamedTuple


class InputSchema(NamedTuple):
    """Schema for the input data for Week 1."""

    flight_details: str = "Flight Details"
    flow_card: str = "Flow Card?"
    bags_checked: str = "Bags Checked"
    meal_type: str = "Meal Type"


class OutputSchema(NamedTuple):
    """Schema fields name for the passenger flight details data that is
    created in the firt challenge.
    """

    id_: str = "id"
    flight_details: str = "flight_details"
    date: str = "date"
    flight_number: str = "flight_number"
    from_: str = "from"
    to: str = "to"
    class_: str = "class"
    price: str = "price"
    has_flow_card: str = "has_flow_card"
    number_of_bags_checked: str = "number_of_bags_checked"
    meal_type: str = "meal_type"
